"""Clause/section-aware chunking.

Instead of fixed-size windows, a new chunk starts whenever the legal structure changes:
Chapter / Part / Schedule / Annex, Article N, Section N(.M), numbered clauses "7.2", GDPR-style
paragraphs "4." inside an Article (-> "Article 83(4)") and recitals "(150)". Long clauses are split
on paragraph boundaries (never mid-sentence) and every piece keeps its clause label, so a citation
can always name page + clause. Tables become their own chunks, labelled with their caption.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .config import SETTINGS
from .ingest import Document, Unit

TOP_RE = re.compile(r"^(CHAPTER|Chapter|PART|Part|SCHEDULE|Schedule|ANNEX|Annex|APPENDIX|Appendix|EXHIBIT|Exhibit)\s+([IVXLC]+|\d+|[A-Z])\b[.:–-]?\s*(.*)")
ART_RE = re.compile(r"^(?:Article|ARTICLE|Art\.)\s+(\d+[a-z]?)\b[.:–-]?\s*(.*)")
SEC_RE = re.compile(r"^(?:Section|SECTION|Clause|CLAUSE|§)\s*(\d+(?:\.\d+)*)\b[.:–-]?\s*(.*)")
NUM_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(?=[A-Z(“\"'])(.*)")
RECITAL_RE = re.compile(r"^\((\d{1,3})\)\s+\S")
CAPTION_RE = re.compile(r"^(Table|TABLE|Figure|FIGURE)\s+([\w.]+)")


@dataclass
class Span:
    page: int
    bbox: tuple[float, float, float, float]
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    doc_title: str
    section: str                 # citation label, e.g. "Section 7.2", "Article 83(4)", "Table 9.1"
    heading_path: str            # e.g. "Section 7 Limitation of Liability > Section 7.2"
    text: str
    kind: str = "text"           # text | table | ocr
    spans: list[Span] = field(default_factory=list)

    @property
    def pages(self) -> list[int]:
        return sorted({s.page for s in self.spans})

    @property
    def page_label(self) -> str:
        p = self.pages
        return f"p. {p[0]}" if len(p) == 1 else f"pp. {p[0]}–{p[-1]}"

    @property
    def index_text(self) -> str:
        """Text used for retrieval: the clause path is prepended so 'Section 7.2' is searchable."""
        return f"{self.doc_title} | {self.heading_path}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        d = dict(d)
        d["spans"] = [Span(s["page"], tuple(s["bbox"]), s["text"]) for s in d["spans"]]
        return cls(**d)


def _is_short_title(u: Unit) -> bool:
    words = u.text.split()
    return 0 < len(words) <= 14 and not NUM_RE.match(u.text) and not u.text.endswith((".", ";", ","))


class _State:
    def __init__(self):
        self.top: str | None = None          # "Schedule B"
        self.major: str | None = None        # "Article 83" | "Section 7"
        self.style: str | None = None        # "article" | "section"
        self.titles: dict[str, str] = {}
        self.label = "Preamble"

    def path(self) -> str:
        parts = []
        if self.top:
            parts.append(f"{self.top} {self.titles.get(self.top, '')}".strip())
        if self.major and self.major != self.top:
            parts.append(f"{self.major} {self.titles.get(self.major, '')}".strip())
        if self.label not in (self.top, self.major):
            parts.append(self.label)
        return " > ".join(parts) or self.label


def _classify(u: Unit, st: _State) -> tuple[str | None, str]:
    """Return (new_label or None, inline title)."""
    t = u.text.strip()
    if m := TOP_RE.match(t):
        label = f"{m.group(1).title()} {m.group(2)}"
        st.top, st.major, st.style = label, None, st.style
        return label, m.group(3)
    if m := ART_RE.match(t):
        label = f"Article {m.group(1)}"
        st.major, st.style = label, "article"
        return label, m.group(2)
    if m := SEC_RE.match(t):
        num = m.group(1)
        st.major, st.style = f"Section {num.split('.')[0]}", "section"
        return f"Section {num}", m.group(2) if "." not in num else ""
    if m := NUM_RE.match(t):
        num = m.group(1)
        if st.style == "article" and st.major and "." not in num:
            return f"{st.major}({num})", ""
        # numbered heading/clause in a contract: "7. LIMITATION OF LIABILITY" / "7.2 The aggregate ..."
        if "." not in num and not (u.bold or len(t.split()) <= 10):
            return None, ""  # a plain numbered list item, not a clause
        st.major, st.style = f"Section {num.split('.')[0]}", "section"
        rest = m.group(2)
        return f"Section {num}", rest if "." not in num and len(rest.split()) <= 10 else ""
    if st.major is None and (m := RECITAL_RE.match(t)):
        return f"Recital {m.group(1)}", ""
    return None, ""


def chunk_document(doc: Document, max_words: int | None = None) -> list[Chunk]:
    max_words = max_words or SETTINGS.max_chunk_words
    st = _State()
    chunks: list[Chunk] = []
    cur_spans: list[Span] = []
    cur_label, cur_path = st.label, st.path()
    pending_caption: str | None = None
    expect_title_for: str | None = None

    def emit(spans: list[Span], label: str, path: str, kind: str = "text"):
        if not spans:
            return
        text = "\n".join(s.text for s in spans)
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}-{len(chunks):04d}", doc_id=doc.doc_id, doc_name=doc.name,
            doc_title=doc.title, section=label, heading_path=path, text=text, kind=kind, spans=list(spans),
        ))

    def flush():
        nonlocal cur_spans
        # heading-only fragments are carried into the next chunk instead of becoming tiny chunks
        is_heading_level = "." not in cur_label and "(" not in cur_label
        if cur_spans and is_heading_level and sum(len(s.text.split()) for s in cur_spans) < 15:
            return
        emit(cur_spans, cur_label, cur_path, "ocr" if all(_k == "ocr" for _k in kinds) else "text")
        cur_spans = []

    kinds: list[str] = []
    footnotes: list[Unit] = []
    for u in doc.units:
        if u.kind == "footnote":
            footnotes.append(u)   # attached to the clause that carries the marker (below)
            continue
        if u.kind == "table":
            label = cur_label
            if pending_caption:
                label = pending_caption
            path = f"{st.path()} > {label}" if label != cur_label else st.path()
            emit([Span(u.page, u.bbox, u.text)], label, path, "table")
            pending_caption = None
            continue

        if expect_title_for and _is_short_title(u):
            st.titles[expect_title_for] = u.text
            cur_path = st.path()
            cur_spans.append(Span(u.page, u.bbox, u.text))
            expect_title_for = None
            continue
        expect_title_for = None

        new_label, inline_title = _classify(u, st)
        if m := CAPTION_RE.match(u.text):
            pending_caption = f"{m.group(1).title()} {m.group(2).rstrip('.:')}"
        if new_label and new_label != cur_label:
            flush()
            cur_label = new_label
            if inline_title and len(inline_title.split()) <= 12:
                st.titles[new_label] = inline_title.rstrip(".")
            elif not inline_title and new_label in (st.major, st.top):
                expect_title_for = new_label
            st.label = new_label
            cur_path = st.path()
            kinds = []
        cur_spans.append(Span(u.page, u.bbox, u.text))
        kinds.append(u.kind)
        if sum(len(s.text.split()) for s in cur_spans) >= max_words:
            emit(cur_spans, cur_label, cur_path)
            cur_spans, kinds = [], []

    if cur_spans:
        emit(cur_spans, cur_label, cur_path, "ocr" if kinds and all(k == "ocr" for k in kinds) else "text")
    _attach_footnotes(chunks, footnotes)
    return chunks


def _attach_footnotes(chunks: list[Chunk], footnotes: list[Unit]) -> None:
    """Footnote '1 The base rate ...' is appended to the clause on the same page whose text carries
    the marker ('... above the base rate1.'), so the footnote is retrieved together with its clause."""
    for fn in footnotes:
        m = re.match(r"\[Footnote\]\s*(\d{1,2}|[*†‡])", fn.text)
        marker = re.escape(m.group(1)) if m else None
        target = None
        for c in chunks:
            if fn.page in c.pages and c.kind != "table":
                if marker and re.search(rf"[A-Za-z)\]]{marker}(?=[\s.,;:]|$)", c.text):
                    target = c
                    break
                target = target or c
        if target:
            target.spans.append(Span(fn.page, fn.bbox, fn.text))
            target.text += "\n" + fn.text
