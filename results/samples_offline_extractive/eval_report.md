# Evaluation report – samples

Generated 2026-11-11 18:42 · 19 questions (10 factual, 5 multi-hop, 4 unanswerable) · seed 42

Backends: dense = `lsa-fallback`, reranker = `none (fallback)`, LLM = `extractive (no LLM)`. Index build: 6.5s for 48 chunks.

## 1. Retrieval (page-level recall@k over answerable questions)

| config | recall@1 | recall@3 | recall@5 | recall@10 | MRR | latency_ms |
|---|---|---|---|---|---|---|
| bm25 | 0.800 | 0.933 | 1.000 | 1.000 | 0.967 | 0.517 |
| dense | 0.800 | 1.000 | 1.000 | 1.000 | 0.967 | 0.987 |
| hybrid | 0.800 | 0.967 | 1.000 | 1.000 | 0.967 | 1.051 |
| hybrid_rerank | 0.800 | 0.967 | 1.000 | 1.000 | 0.967 | 1.011 |

## 2. End-to-end

| config | answer_correct_auto | factual_correct | multihop_correct | citation_precision | citation_recall | region_precision | abstention_accuracy | false_abstention_rate |
|---|---|---|---|---|---|---|---|---|
| hybrid+rerank, planner+verifier (full) | 0.533 | 0.600 | 0.400 | 1.000 | 0.733 | 0.826 | 1.000 | 0.200 |
| dense-only, planner+verifier | 0.533 | 0.600 | 0.400 | 1.000 | 0.733 | 0.826 | 1.000 | 0.200 |
| hybrid+rerank, NO verifier | 0.533 | 0.600 | 0.400 | 1.000 | 0.733 | 0.826 | 1.000 | 0.200 |
| full, context_k=4 (optimisation) | 0.533 | 0.600 | 0.400 | 1.000 | 0.733 | 0.826 | 1.000 | 0.200 |

Answer correctness here is an automatic keyword/number proxy; the manual grade is in `manual_grading.csv`. Citation precision = share of cited pages that are gold evidence pages. Region precision = share of highlighted boxes that overlap the gold evidence region (feature #4).

## 3. Latency and cost (feature #11)

| config | latency_mean_s | latency_p50_s | latency_p95_s | prompt_tokens_per_q | completion_tokens_per_q | context_tokens_per_q | cost_usd_per_q | verifier_retries |
|---|---|---|---|---|---|---|---|---|
| hybrid+rerank, planner+verifier (full) | 0.003 | 0.003 | 0.008 | 0.000 | 0.000 | 384 | 0.000 | 0 |
| dense-only, planner+verifier | 0.003 | 0.003 | 0.007 | 0.000 | 0.000 | 383 | 0.000 | 0 |
| hybrid+rerank, NO verifier | 0.004 | 0.003 | 0.008 | 0.000 | 0.000 | 384 | 0.000 | 0 |
| full, context_k=4 (optimisation) | 0.004 | 0.003 | 0.010 | 0.000 | 0.000 | 255 | 0.000 | 0 |

Optimisation measured: reducing the LLM context from 6 to 4 passages (row *context_k=4*) – compare context tokens / latency with the quality columns above.

## 4. Per-question results (full system)

| id | type | confidence | auto_correct | citation_precision | region_precision | retries | latency_s |
|---|---|---|---|---|---|---|---|
| F01 | factual | HIGH | True | 1.000 | 1.000 | 0 | 0.006 |
| F02 | factual | HIGH | True | 1.000 | 1.000 | 0 | 0.003 |
| F03 | factual | INSUFFICIENT EVIDENCE | False | – | – | 0 | 0.001 |
| F04 | factual | HIGH | True | 1.000 | 0.500 | 0 | 0.004 |
| F05 | factual | INSUFFICIENT EVIDENCE | False | – | – | 0 | 0.004 |
| F06 | factual | HIGH | True | 1.000 | 1.000 | 0 | 0.002 |
| F07 | factual | HIGH | True | 1.000 | 1.000 | 0 | 0.003 |
| F08 | factual | MEDIUM | True | 1.000 | 1.000 | 0 | 0.002 |
| F09 | factual | INSUFFICIENT EVIDENCE | False | – | – | 0 | 0.001 |
| F10 | factual | HIGH | False | 1.000 | 0.500 | 0 | 0.003 |
| M01 | multihop | HIGH | False | 1.000 | 0.250 | 0 | 0.007 |
| M02 | multihop | HIGH | False | 1.000 | 1.000 | 0 | 0.006 |
| M03 | multihop | HIGH | False | 1.000 | 1.000 | 0 | 0.003 |
| M04 | multihop | HIGH | True | 1.000 | 1.000 | 0 | 0.009 |
| M05 | multihop | HIGH | True | 1.000 | 0.667 | 0 | 0.006 |
| U01 | unanswerable | INSUFFICIENT EVIDENCE | True | – | – | 0 | 0.001 |
| U02 | unanswerable | INSUFFICIENT EVIDENCE | True | – | – | 0 | 0.001 |
| U03 | unanswerable | INSUFFICIENT EVIDENCE | True | – | – | 0 | 0.002 |
| U04 | unanswerable | INSUFFICIENT EVIDENCE | True | – | – | 0 | 0.001 |

