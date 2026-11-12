"""Evaluation harness.

  python -m eval.run_eval --questions eval/questions_sample.json            # synthetic bundle
  python -m eval.run_eval --questions eval/questions_gdpr.json --docs data/gdpr

Reports (written to results/<bundle>/):
  * retrieval recall@k / MRR per configuration: bm25, dense, hybrid (RRF), hybrid + cross-encoder rerank
  * end-to-end: answer correctness (auto keyword proxy + manual CSV), citation precision/recall,
    bounding-box region precision, abstention accuracy on unanswerable questions, false-abstention rate,
    latency (mean/p50/p95), tokens per question
  * mandatory comparison: dense-only vs hybrid (and no-retrieval vs retrieval when an LLM is available)
  * verifier on/off (planner+verifier loop), and one measured optimisation (context size 6 -> 4)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lexisync.answer import QAPipeline  # noqa: E402
from lexisync.config import SETTINGS, set_seed  # noqa: E402
from lexisync.llm import get_llm  # noqa: E402
from lexisync.metrics import (citation_precision, citation_recall, keyword_correct, nanmean, pct,  # noqa: E402
                              recall_at_k, reciprocal_rank, region_precision, resolve_gold)
from lexisync.retrieval import Retriever  # noqa: E402
from lexisync.store import Case  # noqa: E402

warnings.filterwarnings("ignore")
KS = (1, 3, 5, 10)


def build_case(docs_dir: Path, work: Path) -> Case:
    if work.exists():
        shutil.rmtree(work)
    case = Case.create(f"eval-{docs_dir.name}", root=work)
    for f in sorted(docs_dir.glob("*.pdf")):
        case.add_pdf(f)
    info = case.build_index()
    logging.info("index: %s", info)
    return case


def retrieval_eval(case: Case, questions: list[dict], golds: dict) -> list[dict]:
    r: Retriever = case.retriever
    rows = []
    configs = [(m, None) for m in Retriever.MODES]
    if r.cross is not None:
        configs += [("hybrid_rerank", 10), ("hybrid_rerank", 50)]
    for mode, depth in configs:
        recs = {k: [] for k in KS}
        mrr, lat = [], []
        for q in questions:
            if q["type"] == "unanswerable":
                continue
            t0 = time.perf_counter()
            hits = r.retrieve(q["question"], k=max(KS), mode=mode, rerank_depth=depth)
            lat.append(time.perf_counter() - t0)
            ranked = [{(h.chunk.doc_id, p) for p in h.chunk.pages} for h in hits]
            for k in KS:
                recs[k].append(recall_at_k(ranked, golds[q["id"]], k))
            mrr.append(reciprocal_rank(ranked, golds[q["id"]]))
        name = mode if depth is None else f"{mode} (depth {depth})"
        rows.append({"config": name, **{f"recall@{k}": nanmean(v) for k, v in recs.items()},
                     "MRR": nanmean(mrr), "latency_ms": 1000 * nanmean(lat)})
    return rows


def e2e_eval(case: Case, questions: list[dict], golds: dict, name: str, llm, **kw) -> tuple[dict, list[dict]]:
    no_ret = kw.pop("no_retrieval", False)
    qa = QAPipeline(case, llm=llm, **kw)
    per_q = []
    for q in questions:
        set_seed()
        a = qa.ask(q["question"], no_retrieval=no_ret)
        g = golds[q["id"]]
        cited = [(c.doc_id, p) for cl in a.claims for c in cl.citations for p in c.pages]
        regions = [(c.doc_id, r["page"], tuple(r["bbox"])) for cl in a.claims for c in cl.citations for r in c.regions]
        ctx_tokens = sum(c.get("chars", len(c["text"])) for c in a.context) // 4
        per_q.append({
            "id": q["id"], "type": q["type"], "question": q["question"], "answer": a.answer,
            "confidence": a.confidence, "abstained": a.abstained,
            "auto_correct": (a.abstained if q["type"] == "unanswerable"
                             else (not a.abstained and keyword_correct(a.answer + " " + " ".join(c.text for c in a.claims),
                                                                         q["keywords"]))),
            "citation_precision": citation_precision(cited, g) if q["type"] != "unanswerable" and not a.abstained else float("nan"),
            "citation_recall": citation_recall(cited, g) if q["type"] != "unanswerable" else float("nan"),
            "region_precision": region_precision(regions, g) if q["type"] != "unanswerable" and not a.abstained else float("nan"),
            "citations": [c.label for cl in a.claims for c in cl.citations],
            "latency_s": a.timings.get("total_s", 0.0), "prompt_tokens": a.tokens["prompt"],
            "completion_tokens": a.tokens["completion"], "context_tokens_est": ctx_tokens,
            "retries": sum(1 for e in a.retry_log if e["event"] == "draft-rejected"),
            "retry_log": a.retry_log, "conflicts": a.conflicts,
        })
    ans = [r for r in per_q if r["type"] != "unanswerable"]
    una = [r for r in per_q if r["type"] == "unanswerable"]
    lat = [r["latency_s"] for r in per_q]
    summary = {
        "config": name, "backend": qa.backend,
        "answer_correct_auto": nanmean([float(r["auto_correct"]) for r in ans]),
        "factual_correct": nanmean([float(r["auto_correct"]) for r in ans if r["type"] == "factual"]),
        "multihop_correct": nanmean([float(r["auto_correct"]) for r in ans if r["type"] == "multihop"]),
        "citation_precision": nanmean([r["citation_precision"] for r in ans]),
        "citation_recall": nanmean([r["citation_recall"] for r in ans]),
        "region_precision": nanmean([r["region_precision"] for r in ans]),
        "abstention_accuracy": nanmean([float(r["abstained"]) for r in una]),
        "false_abstention_rate": nanmean([float(r["abstained"]) for r in ans]),
        "latency_mean_s": nanmean(lat), "latency_p50_s": pct(lat, 50), "latency_p95_s": pct(lat, 95),
        "prompt_tokens_per_q": nanmean([r["prompt_tokens"] for r in per_q]),
        "completion_tokens_per_q": nanmean([r["completion_tokens"] for r in per_q]),
        "context_tokens_per_q": nanmean([r["context_tokens_est"] for r in per_q]),
        "verifier_retries": sum(r["retries"] for r in per_q),
    }
    return summary, per_q


def fmt(v) -> str:
    if isinstance(v, float):
        return "–" if v != v else f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}"
    return str(v)


def table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions_sample.json")
    ap.add_argument("--docs", default=None, help="folder with the PDFs (default: 'bundle' in the questions file)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--llm", default=None, help="ollama | openai | extractive (default: LEXI_LLM_BACKEND)")
    ap.add_argument("--price-in", type=float, default=0.0, help="USD per 1M prompt tokens (0 for local)")
    ap.add_argument("--price-out", type=float, default=0.0, help="USD per 1M completion tokens")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    set_seed()

    spec = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    docs_dir = ROOT / (args.docs or spec["bundle"])
    if not any(docs_dir.glob("*.pdf")):
        sys.exit(f"No PDFs in {docs_dir}")
    out = Path(args.out or ROOT / "results" / docs_dir.name)
    out.mkdir(parents=True, exist_ok=True)
    questions = spec["questions"]

    t0 = time.perf_counter()
    case = build_case(docs_dir, out / "_case")
    index_s = time.perf_counter() - t0
    golds = {q["id"]: resolve_gold(q["gold"], case.documents, case.chunks) for q in questions}
    unresolved = [q["id"] for q in questions for g in golds[q["id"]] if not g["pages"]]
    if unresolved:
        logging.warning("gold evidence not found for: %s", unresolved)

    llm = get_llm(args.llm)
    r = case.retriever
    print(f"Index: {len(case.chunks)} chunks in {index_s:.1f}s | dense={r.dense.backend} | rerank={r.rerank_backend} "
          f"| LLM={llm.name if llm else 'extractive'}")

    ret_rows = retrieval_eval(case, questions, golds)
    print(table(ret_rows, ["config", "recall@1", "recall@3", "recall@5", "recall@10", "MRR", "latency_ms"]))

    configs = [("hybrid+rerank, planner+verifier (full)", dict(mode="hybrid_rerank")),
               ("dense-only, planner+verifier", dict(mode="dense")),
               ("hybrid+rerank, NO verifier", dict(mode="hybrid_rerank", use_verifier=False)),
               ("full, context_k=4 (optimisation)", dict(mode="hybrid_rerank", context_k=4))]
    if llm is not None:
        configs.append(("NO retrieval (LLM only)", dict(no_retrieval=True)))
    summaries, details = [], {}
    for name, kw in configs:
        s, per_q = e2e_eval(case, questions, golds, name, llm, **kw)
        s["cost_usd_per_q"] = (s["prompt_tokens_per_q"] * args.price_in + s["completion_tokens_per_q"] * args.price_out) / 1e6
        summaries.append(s)
        details[name] = per_q
        print(f"  {name}: correct={fmt(s['answer_correct_auto'])} cite_p={fmt(s['citation_precision'])} "
              f"abstain_acc={fmt(s['abstention_accuracy'])} p50={fmt(s['latency_p50_s'])}s")

    # ----- write artefacts
    (out / "retrieval.json").write_text(json.dumps(ret_rows, indent=1), encoding="utf-8")
    (out / "e2e_summary.json").write_text(json.dumps(summaries, indent=1), encoding="utf-8")
    (out / "e2e_details.json").write_text(json.dumps(details, indent=1, default=str), encoding="utf-8")
    full = details[configs[0][0]]
    with open(out / "manual_grading.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "question", "answer", "citations", "confidence", "auto_correct",
                    "manual_correct (1/0)", "notes"])
        for q in full:
            w.writerow([q["id"], q["type"], q["question"], q["answer"], "; ".join(q["citations"]), q["confidence"],
                        int(bool(q["auto_correct"])), "", ""])

    md = [f"# Evaluation report – {docs_dir.name}", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} · {len(questions)} questions "
          f"({sum(q['type'] == 'factual' for q in questions)} factual, "
          f"{sum(q['type'] == 'multihop' for q in questions)} multi-hop, "
          f"{sum(q['type'] == 'unanswerable' for q in questions)} unanswerable) · seed {SETTINGS.seed}", "",
          f"Backends: dense = `{r.dense.backend}`, reranker = `{r.rerank_backend}`, "
          f"LLM = `{llm.name if llm else 'extractive (no LLM)'}`. Index build: {index_s:.1f}s for "
          f"{len(case.chunks)} chunks.", "",
          "## 1. Retrieval (page-level recall@k over answerable questions)", "",
          table(ret_rows, ["config", "recall@1", "recall@3", "recall@5", "recall@10", "MRR", "latency_ms"]), "",
          "## 2. End-to-end", "",
          table(summaries, ["config", "answer_correct_auto", "factual_correct", "multihop_correct",
                            "citation_precision", "citation_recall", "region_precision", "abstention_accuracy",
                            "false_abstention_rate"]), "",
          "Answer correctness here is an automatic keyword/number proxy; the manual grade is in "
          "`manual_grading.csv`. Citation precision = share of cited pages that are gold evidence pages. "
          "Region precision = share of highlighted boxes that overlap the gold evidence region (feature #4).", "",
          "## 3. Latency and cost (feature #11)", "",
          table(summaries, ["config", "latency_mean_s", "latency_p50_s", "latency_p95_s", "prompt_tokens_per_q",
                            "completion_tokens_per_q", "context_tokens_per_q", "cost_usd_per_q", "verifier_retries"]), "",
          "Optimisation measured: reducing the LLM context from 6 to 4 passages (row *context_k=4*) – compare "
          "context tokens / latency with the quality columns above.", "",
          "## 4. Per-question results (full system)", "",
          table(full, ["id", "type", "confidence", "auto_correct", "citation_precision", "region_precision",
                       "retries", "latency_s"]), ""]
    rejected = [(q["id"], e) for q in full for e in q["retry_log"] if e["event"] in ("draft-rejected", "step-cap")]
    if rejected:
        md += ["## 5. Verifier retry log (planner + verifier loop, feature #5)", ""]
        md += [f"- **{qid}** step {e['step']}: {e['event']} – {e['detail']}; "
               + "; ".join(f"“{x['claim'][:80]}” ({x['reason']})" for x in e.get("rejected", [])) for qid, e in rejected]
    (out / "eval_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    shutil.rmtree(out / "_case", ignore_errors=True)
    print(f"\nReport written to {out / 'eval_report.md'}")


if __name__ == "__main__":
    main()
