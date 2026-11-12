from lexisync.chunking import chunk_document
from lexisync.ingest import ingest_pdf


def test_headers_footers_removed_and_two_columns(sample_dir):
    doc = ingest_pdf(sample_dir / "01_master_services_agreement.pdf")
    assert all(p.removed for p in doc.pages), "running header / page number should be removed on every page"
    assert not any("Page 2" == u.text for u in doc.units)
    assert doc.pages[1].columns == 2
    # two-column reading order: definitions stay alphabetical (left column fully before right column)
    defs = [u.text for u in doc.units if u.page == 2 and u.text.startswith("“")]
    assert defs[0].startswith("“Affiliate") and defs[-1].startswith("“Losses")


def test_scanned_page_flagged_and_ocr(sample_dir):
    doc = ingest_pdf(sample_dir / "02_amendment_no1.pdf")
    last = doc.pages[-1]
    assert last.scanned
    if last.ocr_used:  # tesseract installed
        assert any("Jensen" in u.text for u in doc.units if u.kind == "ocr")


def test_clause_aware_labels(case):
    labels = {c.section for c in case.chunks}
    for expected in ["Section 7.2", "Section 9.1", "Table 9.1", "Article 4(1)", "Article 5(1)", "Section 2.2"]:
        assert expected in labels, expected
    table = next(c for c in case.chunks if c.section == "Table 9.1")
    assert table.kind == "table" and "EUR 3,500,000" in table.text and "Intellectual property" in table.text


def test_footnote_attached_to_marker_clause(case):
    c = next(c for c in case.chunks if c.section == "Section 4.2")
    assert "Bundesbank" in c.text


def test_chunks_are_not_fixed_size(case):
    sizes = {len(c.text.split()) for c in case.chunks}
    assert len(sizes) > 10  # boundaries follow clauses, not a fixed window


def test_every_chunk_has_page_and_bbox(case):
    for c in case.chunks:
        assert c.pages and all(len(s.bbox) == 4 for s in c.spans)
