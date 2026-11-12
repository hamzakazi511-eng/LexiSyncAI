"""Hybrid retrieval: BM25 (exact clause numbers such as "7.2", "83(4)") + dense transformer embeddings,
fused with Reciprocal Rank Fusion and reranked by a cross-encoder.

Every model is loaded lazily and cached. If a model cannot be downloaded (offline machine, firewall),
the component degrades gracefully and reports which backend it actually used:
  dense   -> LSA (TF-IDF + truncated SVD) vectors
  rerank  -> skipped (fused order kept); evidence score falls back to lexical IDF coverage
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .config import SETTINGS

log = logging.getLogger(__name__)

STOP = set("""a an the of to in on for and or by with from at as is are was were be been being this that these those
it its their his her our your which who whom what when where why how does do did can could shall should will would may
might must not no any all each other such than then there here into under over about between per up out if so also
only same both either neither within without upon whether i me my we us you he she they them under above below more
most less least much many very just has have had having one two""".split())
QUESTION_WORDS = {"what", "which", "who", "whom", "when", "where", "why", "how", "much", "many", "long", "does", "exceed",
                  "maximum", "minimum", "total", "under", "according", "document", "documents", "agreement", "contract",
                  "party", "parties", "state", "states", "say", "says", "apply", "applies", "current", "currently",
                  "give", "giv", "need", "needed", "take", "taking", "account", "before", "after", "compare", "compar",
                  "original", "originally", "long", "often", "quickly", "happen", "happens", "kind", "kinds", "type",
                  "types", "case", "cases", "new", "adding", "add", "receive", "get", "make", "made", "use", "used"}
# tiny legal query-expansion map (used for the missing-term check and lexical evidence)
SYNONYMS = {"guarantee": ["ensure", "warrant"], "cap": ["aggregate", "limit", "exceed"], "erase": ["delete"],
            "delete": ["erase", "erasure"], "fine": ["penalty"], "sign": ["signatory", "signatories"],
            "terminate": ["termination"], "jurisdiction": ["court"]}

TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*(?:\(\d+\))?|[a-zà-ÿ]+", re.I)


def stem(w: str) -> str:
    """Light suffix stripper: deleted/delete/deletes -> 'delet', increased/increase -> 'increas'."""
    w = w.lower()
    for suf in ("ings", "ing", "ies", "ied", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            w = w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
            break
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    """BM25 tokens. Clause numbers are kept whole ("7.2", "83(4)") *and* split ("83", "4")."""
    out = []
    for tok in TOKEN_RE.findall(text):
        t = tok.lower()
        if t[0].isdigit():
            t = t.replace(",", "")
            out.append(t)
            if "(" in t or "." in t:
                out.extend(p for p in re.split(r"[().]", t) if p)
        elif t not in STOP and len(t) > 1:
            out.append(stem(t))
    return out


QUESTION_WORDS |= {stem(w) for w in QUESTION_WORDS}
SYNONYMS = {stem(k): v for k, v in SYNONYMS.items()}

DOC_REF_RE = re.compile(r"\b(?:the|this|that|a|an|its|their|our|your)\s+((?:[A-Za-z-]+\s+){1,2})(agreement|contract|addendum|amendment|policy|act|regulation|"
                        r"directive|lease|licen[cs]e|treaty|deed|memorandum|charter|bylaws|terms)\b", re.I)


GENERIC_ADJ = {"whole", "entire", "main", "underlying", "relevant", "present", "existing", "new", "old", "signed",
               "said", "applicable", "above", "same", "first", "second", "third", "last", "latest", "final", "full",
               "written", "legal", "binding", "former", "previous", "prior", "later", "uploaded", "given"}


def content_terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in QUESTION_WORDS and not t[0].isdigit()]


# --------------------------------------------------------------------------------------------- models
@lru_cache(maxsize=2)
def _load_embedder(name: str):
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(name, device="cpu")
    except Exception as exc:
        log.warning("Dense model '%s' unavailable (%s) -> LSA fallback", name, str(exc)[:120])
        return None


@lru_cache(maxsize=2)
def _load_cross_encoder(name: str):
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(name, device="cpu")
    except Exception as exc:
        log.warning("Cross-encoder '%s' unavailable (%s) -> rerank disabled", name, str(exc)[:120])
        return None


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.tokens = [tokenize(c.index_text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokens)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


class DenseIndex:
    def __init__(self, chunks: list[Chunk], model_name: str | None = None, embeddings: np.ndarray | None = None):
        self.model_name = model_name or SETTINGS.embed_model
        texts = [c.index_text for c in chunks]
        self.model = _load_embedder(self.model_name)
        if self.model is not None:
            self.backend = self.model_name
            self.emb = embeddings if embeddings is not None else self.model.encode(
                texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        else:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.backend = "lsa-fallback"
            self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
            X = self.vec.fit_transform(texts)
            n = max(2, min(128, X.shape[0] - 1, X.shape[1] - 1))
            self.svd = TruncatedSVD(n_components=n, random_state=SETTINGS.seed).fit(X)
            self.emb = self._norm(self.svd.transform(X))

    @staticmethod
    def _norm(m: np.ndarray) -> np.ndarray:
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

    def encode_query(self, query: str) -> np.ndarray:
        if self.model is not None:
            return self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        return self._norm(self.svd.transform(self.vec.transform([query])))[0]

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        sims = self.emb @ self.encode_query(query)
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx]


def rrf(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_r 1 / (k + rank_r(d)). Scale-free, so BM25 and cosine mix safely."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


@dataclass
class Hit:
    chunk: Chunk
    score: float                    # final ranking score
    evidence: float                 # calibrated [0,1] "does this chunk answer the query" score
    ranks: dict = field(default_factory=dict)


class Retriever:
    MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray | None = None):
        self.chunks = chunks
        self.bm25 = BM25Index(chunks)
        self.dense = DenseIndex(chunks, embeddings=embeddings)
        self.cross = _load_cross_encoder(SETTINGS.rerank_model)
        self.rerank_backend = SETTINGS.rerank_model if self.cross is not None else "none (fallback)"
        # document frequencies for the lexical evidence score / missing-term check
        self.df: dict[str, int] = {}
        for toks in self.bm25.tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.N = len(chunks)

    # ---------------------------------------------------------------- evidence signals
    def idf(self, term: str) -> float:
        return math.log(1 + (self.N + 1) / (self.df.get(term, 0) + 0.5))

    def _known(self, term: str) -> bool:
        return term in self.df or any(stem(s) in self.df for s in SYNONYMS.get(term, []))

    def missing_terms(self, query: str) -> list[str]:
        """Salient query terms that occur in *no* indexed chunk (e.g. 'subcontractor')."""
        return sorted({t for t in content_terms(query) if not self._known(t) and len(t) > 3})

    def unknown_document(self, query: str) -> str | None:
        """'the subcontractor agreement' -> 'subcontractor agreement' if a word naming the document occurs in
        none of the indexed documents, i.e. the question asks about a document that is not in the case."""
        for m in DOC_REF_RE.finditer(query):
            words = [w for w in m.group(1).split() if w.lower() not in STOP | QUESTION_WORDS | GENERIC_ADJ]
            if not words:
                continue
            name_terms = [t for w in words for t in tokenize(w) if not t[0].isdigit()]
            if name_terms and any(not self._known(t) for t in name_terms):
                return f"{' '.join(words)} {m.group(2)}".strip()
        return None

    def lexical_evidence(self, query: str, text: str) -> float:
        q = set(content_terms(query))
        if not q:
            return 0.0
        doc = set(tokenize(text))
        total = sum(self.idf(t) for t in q)
        return sum(self.idf(t) for t in q if t in doc or any(stem(s) in doc for s in SYNONYMS.get(t, []))) / total

    # ---------------------------------------------------------------- retrieval
    def retrieve(self, query: str, k: int = 10, mode: str = "hybrid_rerank", rerank_depth: int | None = None) -> list[Hit]:
        assert mode in self.MODES, mode
        depth = SETTINGS.first_stage_k
        ranks: dict[int, dict] = {}
        bm = self.bm25.search(query, depth) if mode != "dense" else []
        de = self.dense.search(query, depth) if mode != "bm25" else []
        for r, (i, _) in enumerate(bm):
            ranks.setdefault(i, {})["bm25"] = r + 1
        for r, (i, _) in enumerate(de):
            ranks.setdefault(i, {})["dense"] = r + 1

        if mode == "bm25":
            fused = bm
        elif mode == "dense":
            fused = de
        else:
            fused = rrf([bm, de], SETTINGS.rrf_k)

        if mode == "hybrid_rerank" and self.cross is not None:
            cand = fused[: rerank_depth or SETTINGS.rerank_depth]
            logits = self.cross.predict([(query, self.chunks[i].index_text) for i, _ in cand], show_progress_bar=False)
            probs = 1 / (1 + np.exp(-np.asarray(logits, dtype=float)))
            order = np.argsort(-probs)
            hits = []
            for o in order[:k]:
                i = cand[o][0]
                lex = self.lexical_evidence(query, self.chunks[i].index_text)
                hits.append(Hit(self.chunks[i], float(probs[o]), float(0.7 * probs[o] + 0.3 * lex),
                                {**ranks.get(i, {}), "rerank": len(hits) + 1}))
            return hits

        hits = []
        for i, s in fused[:k]:
            lex = self.lexical_evidence(query, self.chunks[i].index_text)
            hits.append(Hit(self.chunks[i], float(s), lex, ranks.get(i, {})))
        return hits
