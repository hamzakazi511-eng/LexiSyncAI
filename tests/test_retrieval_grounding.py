import numpy as np

from lexisync.anomaly import robust_z
from lexisync.grounding import numbers, numeric_support
from lexisync.metrics import keyword_correct, recall_at_k, region_precision
from lexisync.retrieval import rrf, tokenize


def test_tokenizer_keeps_clause_numbers():
    toks = tokenize("Article 83(4) and Section 7.2 apply")
    assert "83(4)" in toks and "83" in toks and "4" in toks
    assert "7.2" in toks and "7" in toks


def test_rrf_rewards_agreement():
    fused = rrf([[(1, 9.0), (2, 5.0), (3, 1.0)], [(2, 0.9), (1, 0.8), (4, 0.1)]])
    assert {fused[0][0], fused[1][0]} == {1, 2}
    assert fused[-1][0] in (3, 4)


def test_bm25_finds_exact_clause(case):
    hits = case.retriever.retrieve("Section 7.2", k=3, mode="bm25")
    assert any(h.chunk.section == "Section 7.2" for h in hits)


def test_hybrid_retrieves_table_for_indemnity_total(case):
    hits = case.retriever.retrieve("total indemnity obligations Section 9", k=5, mode="hybrid")
    assert any(h.chunk.section == "Table 9.1" for h in hits)


def test_missing_term_detection(case):
    assert "subcontractor" in case.retriever.missing_terms("governing law of the subcontractor agreement")


def test_number_normalisation():
    assert numbers("EUR 2.0M") == numbers("EUR 2,000,000") == {2_000_000.0}
    assert 2_000_000.0 in numbers("two million euros")
    assert numbers("Section 7.2 on p. 14, Article 83(4)") == set()


def test_numeric_support_allows_derived_values():
    ok, _ = numeric_support("increase of EUR 150,000", "EUR 1,350,000 ... EUR 1,200,000")
    assert ok
    bad, missing = numeric_support("cap is EUR 3,000,000", "shall not exceed EUR 2,000,000")
    assert not bad and missing == [3_000_000.0]


def test_metrics():
    gold = [{"pages": {("d", 4)}, "bboxes": [("d", 4, (0, 0, 100, 20))]}]
    assert recall_at_k([{("d", 1)}, {("d", 4)}], gold, 1) == 0.0
    assert recall_at_k([{("d", 1)}, {("d", 4)}], gold, 2) == 1.0
    assert region_precision([("d", 4, (0, 5, 100, 15)), ("d", 4, (0, 300, 100, 320))], gold) == 0.5
    assert keyword_correct("The cap is EUR 2.0M", ["2,000,000|2 million"])
    assert not keyword_correct("The cap is EUR 2.5M", ["2,000,000"])


def test_autoencoder_flags_outlier_page(case):
    figure_page = next(d for d in case.documents if d.name.startswith("02")).pages[1]
    assert figure_page.anomalous, "the chart-only page should have a high reconstruction error"
    z = robust_z(np.array([1.0, 1.1, 0.9, 1.0, 5.0]))
    assert z[-1] > 3 and abs(z[0]) < 1


def test_unknown_document_detection(case):
    r = case.retriever
    assert r.unknown_document("What is the governing law of the subcontractor agreement?") == "subcontractor agreement"
    assert r.unknown_document("What is the notice period in the employment contract?") == "employment contract"
    for ok in ["Can the customer cancel the contract early?", "What changed in Amendment No. 1?",
               "Which law governs the Master Services Agreement?", "Is the whole agreement governed by German law?",
               "What does the Data Processing Addendum say about audits?"]:
        assert r.unknown_document(ok) is None, ok
