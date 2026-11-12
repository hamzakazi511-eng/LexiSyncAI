"""PDF ingestion: reading order, header/footer/footnote handling, tables, scanned-page detection + OCR.

Output is a list of `Unit`s (paragraph-level blocks with page number and bounding box), which the
chunker groups into clause-aware chunks. Bounding boxes are kept so citations can be highlighted
on the page image (extended feature #4).
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

from .config import SETTINGS

log = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]

HEADER_ZONE = 0.08   # top 8 % of the page
FOOTER_ZONE = 0.08   # bottom 8 % of the page
MIN_TEXT_CHARS = 25  # fewer extractable characters than this + a big image => scanned page


@dataclass
class Unit:
    text: str
    page: int                      # 1-based
    bbox: BBox
    kind: str = "text"             # text | heading | table | footnote | ocr
    font_size: float = 0.0
    bold: bool = False


@dataclass
class PageInfo:
    number: int
    width: float
    height: float
    n_chars: int = 0
    n_blocks: int = 0
    n_tables: int = 0
    n_images: int = 0
    columns: int = 1
    scanned: bool = False
    ocr_used: bool = False
    ocr_confidence: float | None = None
    removed: list[str] = field(default_factory=list)   # header/footer lines removed
    anomaly_score: float | None = None                 # filled by the page autoencoder
    anomalous: bool = False


@dataclass
class Document:
    doc_id: str
    name: str
    path: str
    title: str
    pages: list[PageInfo]
    units: list[Unit]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(
            doc_id=d["doc_id"], name=d["name"], path=d["path"], title=d["title"],
            pages=[PageInfo(**p) for p in d["pages"]],
            units=[Unit(**{**u, "bbox": tuple(u["bbox"])}) for u in d["units"]],
        )


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _norm_repeating(text: str) -> str:
    """Normalise a header/footer candidate so 'Page 3 of 9' == 'Page 7 of 9'."""
    return re.sub(r"\d+", "#", text.strip().lower())


def _block_text_and_style(block: dict) -> tuple[str, float, bool]:
    lines, sizes, bold_chars, all_chars = [], [], 0, 0
    for line in block.get("lines", []):
        spans = line.get("spans", [])
        txt = "".join(s["text"] for s in spans)
        if txt.strip():
            lines.append(txt.strip())
        for s in spans:
            n = len(s["text"].strip())
            sizes.extend([s["size"]] * max(n, 1))
            all_chars += n
            if "bold" in s.get("font", "").lower() or (s.get("flags", 0) & 16):
                bold_chars += n
    text = " ".join(lines)
    text = re.sub(r"(\w)- (\w)", r"\1\2", text)  # de-hyphenate line breaks
    text = re.sub(r"\s+", " ", text).strip()
    size = float(sorted(sizes)[len(sizes) // 2]) if sizes else 0.0
    return text, size, all_chars > 0 and bold_chars / all_chars > 0.6


def _rect_overlap(a: BBox, b: BBox) -> float:
    """Fraction of `a` covered by `b`."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) or 1.0
    return (x1 - x0) * (y1 - y0) / area


def reading_order(blocks: list[dict], page_width: float) -> tuple[list[dict], int]:
    """Column-aware reading order.

    Full-width blocks (titles, section headings spanning the page) split the page into horizontal
    bands. Inside a band, the left column is read top-to-bottom before the right column. Plain
    y-then-x sorting would interleave the two columns line by line.
    """
    mid = page_width / 2
    blocks = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    out, band, columns = [], [], 1

    def flush():
        nonlocal columns
        left = [b for b in band if (b["bbox"][0] + b["bbox"][2]) / 2 < mid]
        right = [b for b in band if (b["bbox"][0] + b["bbox"][2]) / 2 >= mid]
        if left and right:
            columns = 2
        out.extend(sorted(left, key=lambda b: b["bbox"][1]))
        out.extend(sorted(right, key=lambda b: b["bbox"][1]))
        band.clear()

    for b in blocks:
        width = b["bbox"][2] - b["bbox"][0]
        crosses_mid = b["bbox"][0] < mid - 10 and b["bbox"][2] > mid + 10
        if width > 0.55 * page_width or crosses_mid:
            flush()
            out.append(b)
        else:
            band.append(b)
    flush()
    return out, columns


def _table_to_markdown(rows: list[list]) -> str:
    rows = [[re.sub(r"\s+", " ", str(c or "")).strip() for c in r] for r in rows if r]
    rows = [r for r in rows if any(r)]
    # a spanning header cell comes back as text + empty cells: repeat it over the span
    if rows:
        for i in range(1, len(rows[0])):
            if not rows[0][i] and rows[0][i - 1]:
                rows[0][i] = rows[0][i - 1]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _ocr_page(page: pymupdf.Page) -> tuple[list[Unit], float | None]:
    """OCR an image-only page with Tesseract (extended feature #1, fallback path)."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return [], None
    zoom = 3.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception as exc:  # tesseract binary missing
        log.warning("OCR failed on page %s: %s", page.number + 1, exc)
        return [], None
    groups: dict[tuple, list[int]] = {}
    for i, word in enumerate(data["text"]):
        if word.strip() and float(data["conf"][i]) > 0:
            groups.setdefault((data["block_num"][i], data["par_num"][i]), []).append(i)
    units, confs = [], []
    for idx in groups.values():
        words = [data["text"][i] for i in idx]
        confs += [float(data["conf"][i]) for i in idx]
        x0 = min(data["left"][i] for i in idx) / zoom
        y0 = min(data["top"][i] for i in idx) / zoom
        x1 = max(data["left"][i] + data["width"][i] for i in idx) / zoom
        y1 = max(data["top"][i] + data["height"][i] for i in idx) / zoom
        units.append(Unit(" ".join(words), page.number + 1, (x0, y0, x1, y1), kind="ocr"))
    units.sort(key=lambda u: (u.bbox[1], u.bbox[0]))
    return units, (sum(confs) / len(confs) / 100 if confs else None)


# --------------------------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------------------------
def ingest_pdf(path: str | Path, name: str | None = None) -> Document:
    path = Path(path)
    pdf = pymupdf.open(path)
    doc_id = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    name = name or path.name

    raw_pages = []
    for page in pdf:
        d = page.get_text("dict")
        blocks = [b for b in d["blocks"] if b.get("type") == 0]
        raw_pages.append((page, blocks))

    # ---- 1. detect repeating headers / footers across pages
    candidates: Counter = Counter()
    for page, blocks in raw_pages:
        h = page.rect.height
        seen = set()
        for b in blocks:
            y0, y1 = b["bbox"][1], b["bbox"][3]
            if y1 < h * HEADER_ZONE + 15 or y0 > h * (1 - FOOTER_ZONE) - 15:
                t, _, _ = _block_text_and_style(b)
                if t:
                    seen.add(_norm_repeating(t))
        candidates.update(seen)
    n_pages = len(raw_pages)
    repeating = {k for k, c in candidates.items() if n_pages >= 2 and c >= max(2, 0.5 * n_pages)}

    # body font size (median over the document) – used for footnote + heading detection
    all_sizes = []
    for _, blocks in raw_pages:
        for b in blocks:
            all_sizes.append(_block_text_and_style(b)[1])
    body_size = sorted(all_sizes)[len(all_sizes) // 2] if all_sizes else 10.0

    units: list[Unit] = []
    pages: list[PageInfo] = []
    for page, blocks in raw_pages:
        pno = page.number + 1
        w, h = page.rect.width, page.rect.height
        info = PageInfo(number=pno, width=w, height=h)
        info.n_images = len(page.get_images(full=True))

        # ---- 2. tables (extracted as markdown, their text blocks are removed from the flow)
        table_units, table_rects = [], []
        try:
            for t in page.find_tables().tables:
                # read each cell through its bbox: keeps inter-word spaces that extract() can drop
                rows = [[page.get_textbox(pymupdf.Rect(c)).strip() if c else "" for c in r.cells] for r in t.rows]
                md = _table_to_markdown(rows)
                if md.count("\n") >= 2:
                    rect = tuple(t.bbox)
                    table_rects.append(rect)
                    table_units.append(Unit(md, pno, rect, kind="table"))
        except Exception as exc:  # pragma: no cover - table finder is best effort
            log.debug("table detection failed on p%s: %s", pno, exc)
        info.n_tables = len(table_units)

        # ---- 3. reading order
        ordered, info.columns = reading_order(blocks, w)
        body, footnotes = [], []
        for b in ordered:
            text, size, bold = _block_text_and_style(b)
            if not text:
                continue
            bbox = tuple(round(v, 1) for v in b["bbox"])
            in_margin = bbox[3] < h * HEADER_ZONE + 15 or bbox[1] > h * (1 - FOOTER_ZONE) - 15
            if in_margin and (_norm_repeating(text) in repeating or re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", text.lower())):
                info.removed.append(text)
                continue
            if any(_rect_overlap(bbox, r) > 0.6 for r in table_rects):
                continue
            if bbox[1] > h * 0.5 and size < body_size * 0.88 and re.match(r"^(\d{1,2}|[*†‡])\s?\S", text):
                footnotes.append(Unit(f"[Footnote] {text}", pno, bbox, kind="footnote", font_size=size))
                continue
            body.append(Unit(text, pno, bbox, font_size=size, bold=bold))

        # put tables back at their vertical position
        merged = _interleave_tables(body, table_units)
        page_units = merged + footnotes
        info.n_blocks = len(page_units)
        info.n_chars = sum(len(u.text) for u in page_units)

        # ---- 4. scanned page detection + OCR fallback
        if info.n_chars < MIN_TEXT_CHARS and info.n_images > 0:
            info.scanned = True
            if SETTINGS.ocr_enabled:
                ocr_units, conf = _ocr_page(page)
                if ocr_units:
                    info.ocr_used, info.ocr_confidence = True, conf
                    page_units = ocr_units
                    info.n_chars = sum(len(u.text) for u in ocr_units)
        units.extend(page_units)
        pages.append(info)

    title = (pdf.metadata or {}).get("title") or ""
    if not title:
        first = [u for u in units if u.page == 1 and u.kind == "text"]
        title = max(first[:5], key=lambda u: u.font_size).text[:120] if first else name
    pdf.close()
    return Document(doc_id=doc_id, name=name, path=str(path), title=title, pages=pages, units=units)


def _interleave_tables(body: list[Unit], tables: list[Unit]) -> list[Unit]:
    """Insert each table before the first body block that starts below it (keeps reading order)."""
    out, pending = [], sorted(tables, key=lambda t: t.bbox[1])
    for u in body:
        while pending and pending[0].bbox[1] <= u.bbox[1] and pending[0].bbox[0] < u.bbox[2]:
            out.append(pending.pop(0))
        out.append(u)
    return out + pending
