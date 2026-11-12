# Evaluation report – samples

Generated 2026-11-12 01:58 · 19 questions (10 factual, 5 multi-hop, 4 unanswerable) · seed 42

Backends: dense = `sentence-transformers/all-MiniLM-L6-v2`, reranker = `cross-encoder/ms-marco-MiniLM-L-6-v2`, LLM = `ollama:llama3.1:8b`. Index build: 15.2s for 48 chunks.

## 1. Retrieval (page-level recall@k over answerable questions)

| config | recall@1 | recall@3 | recall@5 | recall@10 | MRR | latency_ms |
|---|---|---|---|---|---|---|
| bm25 | 0.800 | 0.933 | 1.000 | 1.000 | 0.967 | 1.154 |
| dense | 0.867 | 0.967 | 1.000 | 1.000 | 1.000 | 13 |
| hybrid | 0.800 | 1.000 | 1.000 | 1.000 | 0.967 | 9.097 |
| hybrid_rerank | 0.867 | 0.933 | 1.000 | 1.000 | 1.000 | 671 |
| hybrid_rerank (depth 10) | 0.867 | 0.933 | 1.000 | 1.000 | 1.000 | 139 |
| hybrid_rerank (depth 50) | 0.867 | 0.933 | 1.000 | 1.000 | 1.000 | 1121 |

## 2. End-to-end

| config | answer_correct_auto | factual_correct | multihop_correct | citation_precision | citation_recall | region_precision | abstention_accuracy | false_abstention_rate |
|---|---|---|---|---|---|---|---|---|
| hybrid+rerank, planner+verifier (full) | 0.800 | 0.900 | 0.600 | 1.000 | 0.900 | 0.833 | 0.750 | 0.000 |
| dense-only, planner+verifier | 0.533 | 0.800 | 0.000 | 1.000 | 0.867 | 0.733 | 0.500 | 0.000 |
| hybrid+rerank, NO verifier | 0.733 | 0.900 | 0.400 | 0.967 | 0.900 | 0.789 | 0.750 | 0.000 |
| full, context_k=4 (optimisation) | 0.667 | 0.800 | 0.400 | 1.000 | 0.867 | 0.867 | 1.000 | 0.000 |
| NO retrieval (LLM only) | 0.133 | 0.200 | 0.000 | – | 0.000 | – | 0.000 | 0.000 |

Answer correctness here is an automatic keyword/number proxy; the manual grade is in `manual_grading.csv`. Citation precision = share of cited pages that are gold evidence pages. Region precision = share of highlighted boxes that overlap the gold evidence region (feature #4).

## 3. Latency and cost (feature #11)

| config | latency_mean_s | latency_p50_s | latency_p95_s | prompt_tokens_per_q | completion_tokens_per_q | context_tokens_per_q | cost_usd_per_q | verifier_retries |
|---|---|---|---|---|---|---|---|---|
| hybrid+rerank, planner+verifier (full) | 12 | 12 | 21 | 1306 | 185 | 500 | 0.000 | 2 |
| dense-only, planner+verifier | 9.646 | 9.441 | 17 | 1280 | 179 | 449 | 0.000 | 2 |
| hybrid+rerank, NO verifier | 8.096 | 8.529 | 9.462 | 923 | 133 | 494 | 0.000 | 0 |
| full, context_k=4 (optimisation) | 12 | 11 | 21 | 1264 | 189 | 395 | 0.000 | 5 |
| NO retrieval (LLM only) | 4.289 | 4.265 | 5.340 | 96 | 147 | 0.000 | 0.000 | 0 |

Optimisation measured: reducing the LLM context from 6 to 4 passages (row *context_k=4*) – compare context tokens / latency with the quality columns above.

## 4. Per-question results (full system)

| id | type | confidence | auto_correct | citation_precision | region_precision | retries | latency_s |
|---|---|---|---|---|---|---|---|
| F01 | factual | HIGH | True | 1.000 | 1.000 | 0 | 11 |
| F02 | factual | HIGH | True | 1.000 | 1.000 | 0 | 13 |
| F03 | factual | MEDIUM | True | 1.000 | 1.000 | 1 | 25 |
| F04 | factual | HIGH | True | 1.000 | 0.500 | 0 | 12 |
| F05 | factual | HIGH | True | 1.000 | 0.333 | 0 | 11 |
| F06 | factual | HIGH | True | 1.000 | 1.000 | 0 | 12 |
| F07 | factual | HIGH | True | 1.000 | 1.000 | 0 | 12 |
| F08 | factual | MEDIUM | True | 1.000 | 0.500 | 0 | 11 |
| F09 | factual | HIGH | True | 1.000 | 1.000 | 0 | 11 |
| F10 | factual | HIGH | False | 1.000 | 0.500 | 0 | 12 |
| M01 | multihop | MEDIUM | True | 1.000 | 1.000 | 1 | 21 |
| M02 | multihop | HIGH | False | 1.000 | 1.000 | 0 | 12 |
| M03 | multihop | LOW | True | 1.000 | 1.000 | 0 | 12 |
| M04 | multihop | MEDIUM | False | 1.000 | 1.000 | 0 | 11 |
| M05 | multihop | HIGH | True | 1.000 | 0.667 | 0 | 14 |
| U01 | unanswerable | INSUFFICIENT EVIDENCE | True | – | – | 0 | 3.605 |
| U02 | unanswerable | INSUFFICIENT EVIDENCE | True | – | – | 0 | 5.066 |
| U03 | unanswerable | MEDIUM | False | – | – | 0 | 13 |
| U04 | unanswerable | INSUFFICIENT EVIDENCE | True | – | – | 0 | 5.719 |

## 5. Verifier retry log (planner + verifier loop, feature #5)

- **F03** step 1: draft-rejected – 1/2 claims supported; “The Supplier guarantees at least 98.5% monthly platform availability in 2025.” (LLM judge: different document)
- **M01** step 1: draft-rejected – 0/1 claims supported; “The liability cap in Section 7 exceeds the total indemnity obligations in Sectio” (LLM judge: The liability cap in Section 7 is EUR 2,000,000, which is lower than the total indemnity obligations in Section 9, which is EUR 3,500,000.)
