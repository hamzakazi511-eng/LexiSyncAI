"""Central configuration. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default):
    val = os.getenv(name)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.lower() in {"1", "true", "yes"}
    return type(default)(val)


@dataclass
class Settings:
    # storage
    data_dir: Path = field(default_factory=lambda: Path(_env("LEXI_DATA_DIR", str(ROOT / "data" / "cases"))))
    seed: int = _env("LEXI_SEED", 42)

    # models (downloaded from Hugging Face on first use; graceful fallback if unavailable)
    embed_model: str = _env("LEXI_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    rerank_model: str = _env("LEXI_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # LLM backend: "ollama" | "openai" | "extractive"
    llm_backend: str = _env("LEXI_LLM_BACKEND", "ollama")
    ollama_url: str = _env("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = _env("LEXI_OLLAMA_MODEL", "llama3.1:8b")
    openai_base_url: str = _env("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    openai_model: str = _env("LEXI_OPENAI_MODEL", "llama-3.1-8b-instant")
    openai_api_key: str = _env("OPENAI_API_KEY", "")
    llm_timeout: int = _env("LEXI_LLM_TIMEOUT", 180)

    # chunking
    max_chunk_words: int = _env("LEXI_MAX_CHUNK_WORDS", 320)

    # retrieval
    first_stage_k: int = _env("LEXI_FIRST_STAGE_K", 30)   # candidates from each retriever
    rerank_depth: int = _env("LEXI_RERANK_DEPTH", 20)     # fused candidates sent to the cross-encoder
    context_k: int = _env("LEXI_CONTEXT_K", 6)            # chunks passed to the LLM
    rrf_k: int = 60

    # abstention / confidence (evidence score in [0, 1])
    abstain_threshold: float = _env("LEXI_ABSTAIN_T", 0.30)          # with cross-encoder
    abstain_threshold_lexical: float = _env("LEXI_ABSTAIN_T_LEX", 0.20)  # fallback: no cross-encoder
    high_conf_threshold: float = _env("LEXI_HIGH_T", 0.70)

    # planner + verifier loop
    max_verify_steps: int = _env("LEXI_MAX_STEPS", 3)

    # OCR
    ocr_enabled: bool = _env("LEXI_OCR", True)

    # page autoencoder
    ae_epochs: int = _env("LEXI_AE_EPOCHS", 60)
    ae_z_threshold: float = _env("LEXI_AE_Z", 2.0)


SETTINGS = Settings()


def set_seed(seed: int | None = None) -> None:
    seed = SETTINGS.seed if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover
        pass
