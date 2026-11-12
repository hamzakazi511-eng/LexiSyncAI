"""Evaluation metrics (pure functions, unit-tested)."""
from __future__ import annotations

import re

import numpy as np

from .grounding import numbers


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("“", '"').replace("”", '"')).strip().lower()


def resolve_gold(gold: list[dict], documents, chunks) -> list[dict]:
    """Turn {'doc': prefix, 'evidence': phrase} / {'doc': prefix, 'section': label} into
    {'doc_id', 'pages': set, 'bboxes': [(page, bbox)]} using the ingested text (works for OCR pages too)."""
    out = []
    for g in gold:
        docs = [d for d in documents if d.name.startswith(g.get("doc", ""))]
        item = {"doc_ids": {d.doc_id for d in docs}, "pages": set(), "bboxes": [], "label": g}
        if g.get("evidence"):
            phrase = _norm(g["evidence"])
            for d in docs:
                for u in d.units:
                    if phrase in _norm(u.text):
                        item["pages"].add((d.doc_id, u.page))
                        item["bboxes"].append((d.doc_id, u.page, u.bbox))
        if g.get("section"):
            sec = g["section"]
            for c in chunks:
                if c.doc_id in item["doc_ids"] and (c.section == sec or c.section.startswith(sec + "(")
                                                    or c.section.startswith(sec + ".")):
                    for p in c.pages:
                        item["pages"].add((c.doc_id, p))
        out.append(item)
    return out


def recall_at_k(ranked_pages: list[set], gold_items: list[dict], k: int) -> float:
    """Fraction of gold evidence items whose (doc, page) appears in the top-k retrieved chunks."""
    if not gold_items:
        return float("nan")
    top = set().union(*ranked_pages[:k]) if ranked_pages[:k] else set()
    return sum(1 for g in gold_items if g["pages"] & top) / len(gold_items)


def reciprocal_rank(ranked_pages: list[set], gold_items: list[dict]) -> float:
    gold = set().union(*(g["pages"] for g in gold_items)) if gold_items else set()
    for i, pages in enumerate(ranked_pages):
        if pages & gold:
            return 1.0 / (i + 1)
    return 0.0


def keyword_correct(answer: str, keywords: list[str]) -> bool:
    """Automatic correctness proxy: every keyword group must be present (numbers are compared numerically)."""
    a = _norm(answer)
    a_nums = numbers(answer)
    for group in keywords:
        ok = False
        for alt in group.split("|"):
            alt_nums = numbers(alt)
            if alt_nums and alt_nums <= a_nums:
                ok = True
            elif _norm(alt) in a:
                ok = True
        if not ok:
            return False
    return True


def citation_precision(cited: list[tuple[str, int]], gold_items: list[dict]) -> float:
    if not cited:
        return float("nan")
    gold = set().union(*(g["pages"] for g in gold_items)) if gold_items else set()
    return sum(1 for c in cited if c in gold) / len(cited)


def citation_recall(cited: list[tuple[str, int]], gold_items: list[dict]) -> float:
    if not gold_items:
        return float("nan")
    cited = set(cited)
    return sum(1 for g in gold_items if g["pages"] & cited) / len(gold_items)


def _overlap(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0) / max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)


def region_precision(regions: list[tuple[str, int, tuple]], gold_items: list[dict], min_overlap: float = 0.3) -> float:
    """Bounding-box citation precision: a predicted region is correct if >= 30 % of it lies on a gold region."""
    if not regions:
        return float("nan")
    gold = [b for g in gold_items for b in g["bboxes"]]
    good = 0
    for doc_id, page, box in regions:
        if any(d == doc_id and p == page and (_overlap(box, gb) >= min_overlap or _overlap(gb, box) >= min_overlap)
               for d, p, gb in gold):
            good += 1
    return good / len(regions)


def nanmean(xs) -> float:
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def pct(xs, q) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")
