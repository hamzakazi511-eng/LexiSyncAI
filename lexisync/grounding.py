"""Evidence grounding helpers.

* numbers(): normalises legal amounts so "EUR 2.0M", "EUR 2,000,000", "2 million" and "two million" all
  become 2000000.0 – the deterministic part of the verifier checks that every number in a claim is
  present in (or arithmetically derivable from) the passages it cites.
* best_regions(): picks the paragraph boxes inside a cited chunk that support a claim, then tightens them
  to the exact matched phrase where possible (extended feature #4, bounding-box citations).
"""
from __future__ import annotations

import re
from itertools import combinations

from .chunking import Chunk, Span
from .retrieval import tokenize

WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
           "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30,
           "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90, "hundred": 100}
SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "millions": 1e6, "bn": 1e9, "b": 1e9,
         "billion": 1e9}

# references that are not quantities: section numbers, page numbers, article numbers, dates, years
REF_RE = re.compile(
    r"(?:Section|Sections|Clause|Article|Art\.|Table|Schedule|Annex|Recital|§|pp?\.)\s*[\w.()\-–]+"
    r"(?:\s*(?:and|,|–|-)\s*[\d.()]+)*|\[C\d+\]|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}|\b(?:19|20)\d{2}\b|\bNo\.\s*\d+|\b\d{4}/\d+\b|\(\d+\)", re.I)
NUM_RE = re.compile(r"(\d+(?:[.,]\d+)*)\s*(%|percent|k|m|mn|bn|b|thousand|millions?|billion)?\b", re.I)


def _to_float(raw: str) -> float | None:
    raw = raw.strip()
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", raw):
        raw = raw.replace(",", "")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", raw):          # German style 2.000.000,00
        raw = raw.replace(".", "").replace(",", ".")
    else:
        raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def numbers(text: str) -> set[float]:
    text = REF_RE.sub(" ", text)
    out: set[float] = set()
    for m in NUM_RE.finditer(text):
        v = _to_float(m.group(1))
        if v is None:
            continue
        unit = (m.group(2) or "").lower()
        if unit in SCALE:
            v *= SCALE[unit]
        out.add(round(v, 4))
    for m in re.finditer(r"\b(" + "|".join(WORDNUM) + r")(?:[\s-](" + "|".join(WORDNUM) + r"))?\s+(thousand|million|billion)?",
                         text, re.I):
        v = WORDNUM[m.group(1).lower()] + (WORDNUM[m.group(2).lower()] if m.group(2) else 0)
        if m.group(3):
            v *= SCALE[m.group(3).lower()]
        out.add(float(v))
    return out


def numeric_support(claim: str, evidence: str) -> tuple[bool, list[float]]:
    """True if every quantity in the claim appears in the evidence, or is a sum / difference / ratio of two
    evidence quantities (e.g. 'an increase of EUR 150,000' from 1,350,000 and 1,200,000)."""
    claim_nums = {n for n in numbers(claim) if n not in (0.0, 1.0)}
    ev = numbers(evidence)
    derived = set(ev)
    for a, b in combinations(sorted(ev), 2):
        derived |= {round(a + b, 4), round(abs(a - b), 4)}
        if a:
            derived.add(round(b / a, 4))
            derived.add(round(100 * (b - a) / a, 2))
    missing = [n for n in claim_nums if not any(abs(n - d) <= max(1e-6, 0.005 * abs(d)) for d in derived)]
    return not missing, missing


def token_overlap(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    return len(ta & tb) / len(ta) if ta else 0.0


def best_regions(claim: str, chunk: Chunk, page_lookup=None, max_regions: int = 2) -> list[dict]:
    """Return [{'page', 'bbox', 'exact'}] – paragraph boxes in the chunk that best support the claim.

    `page_lookup(page_no)` may return an open pymupdf.Page; if given, boxes are tightened to the exact
    occurrence of the claim's key numbers/phrases inside the paragraph box.
    """
    scored = sorted(((token_overlap(claim, s.text) + 0.5 * _num_hit(claim, s), s) for s in chunk.spans),
                    key=lambda x: -x[0])
    chosen: list[Span] = [s for sc, s in scored[:max_regions] if sc >= 0.15] or [scored[0][1]]
    regions = []
    for s in chosen:
        reg = {"page": s.page, "bbox": tuple(s.bbox), "exact": False}
        if page_lookup is not None:
            tight = _tighten(claim, s, page_lookup(s.page))
            if tight:
                reg = {"page": s.page, "bbox": tight, "exact": True}
        regions.append(reg)
    return regions


def _num_hit(claim: str, span: Span) -> float:
    c, s = numbers(claim), numbers(span.text)
    return 1.0 if c and c & s else 0.0


def _tighten(claim: str, span: Span, page) -> tuple | None:
    """Search the page for the literal number strings of the span that also occur in the claim."""
    import pymupdf

    if page is None:
        return None
    clip = pymupdf.Rect(span.bbox) + (-2, -2, 2, 2)
    wanted = numbers(claim)
    rects = []
    for m in NUM_RE.finditer(span.text):
        raw = m.group(0).strip()
        v = _to_float(m.group(1))
        if v is None:
            continue
        unit = (m.group(2) or "").lower()
        if unit in SCALE:
            v *= SCALE[unit]
        if any(abs(v - w) <= 0.005 * max(abs(w), 1) for w in wanted):
            for r in page.search_for(raw, clip=clip):
                rects.append(r)
    if not rects:
        return None
    u = rects[0]
    for r in rects[1:]:
        u |= r
    # widen to the full line(s) so the highlight is readable
    return (clip.x0 + 2, u.y0 - 1, clip.x1 - 2, u.y1 + 1)
