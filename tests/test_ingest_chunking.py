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


def test_gdpr_cross_references_are_not_headings():
    from lexisync.ingest import Document, PageInfo, Unit

    texts = ["Article 83", "General conditions for imposing administrative fines",
             "1. Each supervisory authority shall ensure that the imposition of administrative fines is effective.",
             "2. When deciding whether to impose an administrative fine due regard shall be given to the following:",
             "(j) adherence to approved codes of conduct pursuant to Article 40 or approved certification mechanisms pursuant to",
             "Article 42; and",
             "Article 40 or the certification mechanism referred to in the following paragraph.",
             "4. Infringements of the following provisions shall be subject to administrative fines up to 10 000 000 EUR.",
             "5. Infringements of the following provisions shall be subject to administrative fines up to 20 000 000 EUR.",
             "Article 84", "Penalties",
             "1. Member States shall lay down the rules on other penalties applicable to infringements of this Regulation."]
    units = [Unit(t, 1, (50, 50 + 20 * i, 500, 65 + 20 * i)) for i, t in enumerate(texts)]
    doc = Document("g", "gdpr.pdf", "x", "GDPR", [PageInfo(1, 595, 842)], units)
    labels = {c.section for c in chunk_document(doc)}
    assert "Article 83(4)" in labels and "Article 83(5)" in labels and "Article 84(1)" in labels
    assert not any(l.startswith(("Article 42", "Article 40")) for l in labels), labels
