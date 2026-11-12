"""A *case* = a folder with 2–5 public PDFs, their parsed structure, chunks and index artefacts."""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
import uuid
from functools import cached_property
from pathlib import Path

import numpy as np
import pymupdf
import requests

from .chunking import Chunk, chunk_document
from .config import SETTINGS
from .ingest import Document, ingest_pdf

log = logging.getLogger(__name__)


class Case:
    def __init__(self, case_id: str, root: Path | None = None):
        self.case_id = case_id
        self.dir = Path(root or SETTINGS.data_dir) / case_id
        self.meta_path = self.dir / "case.json"
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8")) if self.meta_path.exists() else {}
        self.documents: list[Document] = [Document.from_dict(d) for d in self.meta.get("documents", [])]
        chunks_path = self.dir / "chunks.json"
        self.chunks: list[Chunk] = ([Chunk.from_dict(c) for c in json.loads(chunks_path.read_text(encoding="utf-8"))]
                                    if chunks_path.exists() else [])
        self._pdfs: dict[str, pymupdf.Document] = {}

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def create(cls, name: str, root: Path | None = None) -> "Case":
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:30] or "case"
        case = cls(f"{slug}-{uuid.uuid4().hex[:6]}", root)
        (case.dir / "docs").mkdir(parents=True, exist_ok=True)
        case.meta = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M"), "documents": []}
        case.save()
        return case

    @staticmethod
    def list_cases(root: Path | None = None) -> list[dict]:
        root = Path(root or SETTINGS.data_dir)
        out = []
        for p in sorted(root.glob("*/case.json")):
            m = json.loads(p.read_text(encoding="utf-8"))
            out.append({"case_id": p.parent.name, "name": m.get("name"), "n_docs": len(m.get("documents", [])),
                        "created": m.get("created")})
        return out

    @property
    def name(self) -> str:
        return self.meta.get("name", self.case_id)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta["documents"] = [d.to_dict() for d in self.documents]
        self.meta_path.write_text(json.dumps(self.meta, indent=1), encoding="utf-8")
        (self.dir / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks]), encoding="utf-8")

    # ------------------------------------------------------------------ documents
    def add_pdf(self, src: str | Path | bytes, filename: str | None = None) -> Path:
        docs_dir = self.dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(src, (bytes, bytearray)):
            dest = docs_dir / _safe_name(filename or "upload.pdf")
            dest.write_bytes(src)
        else:
            dest = docs_dir / _safe_name(filename or Path(src).name)
            shutil.copy(src, dest)
        pymupdf.open(dest).close()  # validate
        return dest

    def add_url(self, url: str) -> Path:
        r = requests.get(url, timeout=60, headers={"User-Agent": "LexiSyncAI/1.0 (academic project)"})
        r.raise_for_status()
        if not r.content[:5] == b"%PDF-":
            raise ValueError("URL did not return a PDF")
        name = Path(url.split("?")[0]).name or "download.pdf"
        return self.add_pdf(r.content, name if name.endswith(".pdf") else name + ".pdf")

    def remove_pdf(self, filename: str) -> None:
        (self.dir / "docs" / filename).unlink(missing_ok=True)

    def pdf_files(self) -> list[Path]:
        return sorted((self.dir / "docs").glob("*.pdf"))

    # ------------------------------------------------------------------ indexing
    def build_index(self, train_autoencoder: bool = True, progress=None) -> dict:
        t0 = time.perf_counter()
        files = self.pdf_files()
        if not 1 <= len(files) <= 5:
            log.warning("The brief expects 2–5 PDFs per case, got %d", len(files))
        self.documents, self.chunks = [], []
        for i, f in enumerate(files):
            if progress:
                progress(i / max(len(files), 1), f"Parsing {f.name}")
            doc = ingest_pdf(f)
            self.documents.append(doc)
            self.chunks.extend(chunk_document(doc))
        timings = {"parse_chunk_s": time.perf_counter() - t0}

        if train_autoencoder and self.documents:
            if progress:
                progress(0.8, "Training page autoencoder")
            t1 = time.perf_counter()
            from .anomaly import score_pages

            score_pages(self.documents)
            timings["autoencoder_s"] = time.perf_counter() - t1

        self.__dict__.pop("retriever", None)
        if progress:
            progress(0.9, "Embedding chunks")
        t2 = time.perf_counter()
        r = self.retriever
        if r.dense.backend != "lsa-fallback":
            np.save(self._emb_path(), r.dense.emb)
        timings["embed_s"] = time.perf_counter() - t2
        self.meta["index"] = {"built": time.strftime("%Y-%m-%d %H:%M"), "n_chunks": len(self.chunks),
                              "dense": r.dense.backend, "rerank": r.rerank_backend, **timings}
        self.save()
        return self.meta["index"]

    def _emb_path(self) -> Path:
        return self.dir / f"emb_{re.sub(r'[^a-zA-Z0-9]+', '_', SETTINGS.embed_model)}.npy"

    @cached_property
    def retriever(self):
        from .retrieval import Retriever

        emb = None
        if self._emb_path().exists():
            emb = np.load(self._emb_path())
            if emb.shape[0] != len(self.chunks):
                emb = None
        return Retriever(self.chunks, embeddings=emb)

    # ------------------------------------------------------------------ pages
    def doc(self, doc_id: str) -> Document:
        return next(d for d in self.documents if d.doc_id == doc_id)

    def pdf(self, doc_id: str) -> pymupdf.Document:
        if doc_id not in self._pdfs:
            self._pdfs[doc_id] = pymupdf.open(self.doc(doc_id).path)
        return self._pdfs[doc_id]

    def page(self, doc_id: str, page_no: int) -> pymupdf.Page:
        return self.pdf(doc_id)[page_no - 1]

    def render_page(self, doc_id: str, page_no: int, regions: list[dict] | None = None, zoom: float = 1.6) -> bytes:
        """PNG of a page with the cited regions drawn as translucent boxes (bounding-box citations)."""
        src = self.pdf(doc_id)
        tmp = pymupdf.open()
        tmp.insert_pdf(src, from_page=page_no - 1, to_page=page_no - 1)
        pg = tmp[0]
        for r in regions or []:
            if r["page"] != page_no:
                continue
            rect = pymupdf.Rect(r["bbox"]) + (-3, -3, 3, 3)
            color = (0.95, 0.55, 0.0) if r.get("exact") else (0.1, 0.45, 0.95)
            pg.draw_rect(rect, color=color, fill=color, fill_opacity=0.18, width=1.6)
        png = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")
        tmp.close()
        return png


def _safe_name(name: str) -> str:
    name = re.sub(r"[^\w.\-]+", "_", Path(name).name)
    return name if name.lower().endswith(".pdf") else name + ".pdf"
