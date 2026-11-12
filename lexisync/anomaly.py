"""Convolutional autoencoder page-anomaly detector (extended feature #8, links to the Autoencoder lecture).

Every page is rendered to a 64x64 grayscale thumbnail. A small conv autoencoder is trained to reconstruct
the pages of the case (unsupervised, no labels). Typical text pages share a layout and are reconstructed
well; pages that look different – scanned images, charts/figures, signature pages, blank or rotated pages –
have a high reconstruction error  L = ||x - x_hat||^2.  We flag pages whose robust z-score
(median / MAD) exceeds a threshold. These pages are shown in the structure view and their evidence gets a
lower confidence, because text extraction on them is less reliable.
"""
from __future__ import annotations

import logging

import numpy as np
import pymupdf
from PIL import Image

from .config import SETTINGS, set_seed
from .ingest import Document

log = logging.getLogger(__name__)
SIZE = 64


def page_thumbnail(page: pymupdf.Page, size: int = SIZE) -> np.ndarray:
    pix = page.get_pixmap(matrix=pymupdf.Matrix(0.4, 0.4), colorspace=pymupdf.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples).resize((size, size), Image.BILINEAR)
    return 1.0 - np.asarray(img, dtype=np.float32) / 255.0  # ink = 1, paper = 0


def _model(latent: int = 16):
    import torch.nn as nn

    class PageAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),    # 32x32
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # 16x16
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),   # 8x8
                nn.Flatten(), nn.Linear(32 * 8 * 8, latent))
            self.decoder = nn.Sequential(
                nn.Linear(latent, 32 * 8 * 8), nn.ReLU(), nn.Unflatten(1, (32, 8, 8)),
                nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
                nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), nn.Sigmoid())

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return PageAE()


def train_and_score(thumbs: np.ndarray, epochs: int | None = None) -> np.ndarray:
    """Train on all thumbnails, return per-page reconstruction MSE."""
    import torch

    set_seed()
    epochs = epochs or SETTINGS.ae_epochs
    x = torch.tensor(thumbs[:, None])
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()
    g = torch.Generator().manual_seed(SETTINGS.seed)
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), 16):
            xb = x[perm[i:i + 16]]
            noisy = (xb + 0.02 * torch.randn(xb.shape, generator=g)).clamp(0, 1)   # denoising AE
            opt.zero_grad()
            loss = loss_fn(model(noisy), xb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return ((model(x) - x) ** 2).mean(dim=(1, 2, 3)).numpy()


def robust_z(errors: np.ndarray) -> np.ndarray:
    med = np.median(errors)
    mad = np.median(np.abs(errors - med)) * 1.4826 + 1e-9
    return (errors - med) / mad


def score_pages(documents: list[Document]) -> list[tuple[str, int, float, bool]]:
    refs, thumbs = [], []
    for d in documents:
        pdf = pymupdf.open(d.path)
        for p in pdf:
            refs.append((d, p.number))
            thumbs.append(page_thumbnail(p))
        pdf.close()
    if len(thumbs) < 4:
        log.info("Too few pages for the autoencoder (%d)", len(thumbs))
        return []
    errors = train_and_score(np.stack(thumbs))
    z = robust_z(errors)
    out = []
    for (d, pidx), e, zz in zip(refs, errors, z):
        info = d.pages[pidx]
        info.anomaly_score = round(float(zz), 2)
        info.anomalous = bool(zz > SETTINGS.ae_z_threshold)
        out.append((d.name, pidx + 1, float(e), info.anomalous))
    return out
