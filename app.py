"""LexiSyncAI – Streamlit UI.   Run:  streamlit run app.py"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

from lexisync.answer import QAPipeline
from lexisync.config import ROOT, SETTINGS, set_seed
from lexisync.llm import get_llm
from lexisync.store import Case

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
set_seed()
st.set_page_config(page_title="LexiSyncAI – Evidence-Grounded Legal Analyst", page_icon="⚖️", layout="wide")

BADGE = {"HIGH": ("#1b7f3b", "HIGH confidence"), "MEDIUM": ("#b7791f", "MEDIUM confidence"),
         "LOW": ("#c05621", "LOW confidence"), "INSUFFICIENT EVIDENCE": ("#9b2c2c", "INSUFFICIENT EVIDENCE · abstained"),
         "UNGROUNDED": ("#4a5568", "UNGROUNDED")}
EXAMPLES = ["Does the liability cap in Section 7 exceed the total indemnity obligations in Section 9?",
            "What is the current aggregate liability cap, taking Amendment No. 1 into account?",
            "Who signed Amendment No. 1 on behalf of Clearpath Software Ltd?",
            "What is the governing law of the subcontractor agreement?"]


@st.cache_resource(show_spinner=False)
def load_case(case_id: str, version: str) -> Case:  # version busts the cache after re-indexing
    return Case(case_id)


@st.cache_resource(show_spinner="Connecting to LLM…")
def load_llm(backend: str):
    return get_llm(backend)


def badge(conf: str) -> str:
    color, text = BADGE.get(conf, ("#4a5568", conf))
    return (f"<span style='background:{color};color:white;padding:4px 12px;border-radius:12px;"
            f"font-weight:600;font-size:0.9rem'>{text}</span>")


# --------------------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("⚖️ LexiSyncAI")
    st.caption("Evidence-grounded legal document analyst")
    cases = Case.list_cases()
    labels = {c["case_id"]: f"{c['name']} ({c['n_docs']} docs)" for c in cases}
    if cases:
        default = st.session_state.get("case_id")
        ids = list(labels)
        idx = ids.index(default) if default in ids else 0
        st.session_state.case_id = st.selectbox("Case", ids, index=idx, format_func=labels.get)
    with st.expander("➕ New case", expanded=not cases):
        name = st.text_input("Case name", placeholder="e.g. Vendor contract review")
        if st.button("Create case", width="stretch", disabled=not name):
            st.session_state.case_id = Case.create(name).case_id
            st.session_state.pop("answers", None)
            st.rerun()

    st.divider()
    st.subheader("Settings")
    backend = st.selectbox("LLM backend", ["ollama", "openai", "extractive"],
                           index=["ollama", "openai", "extractive"].index(SETTINGS.llm_backend),
                           help="ollama = local model; openai = any OpenAI-compatible API (Groq, OpenRouter, OCI); "
                                "extractive = no LLM, sentence selection")
    mode = st.selectbox("Retrieval", ["hybrid_rerank", "hybrid", "dense", "bm25"],
                        help="hybrid_rerank = BM25 + dense (RRF) + cross-encoder")
    use_verifier = st.toggle("Planner + verifier loop", value=True)
    llm = load_llm(backend)
    st.caption(f"LLM in use: **{llm.name if llm else 'extractive (no LLM)'}**")

if "case_id" not in st.session_state:
    st.info("Create a case in the sidebar to start.")
    st.stop()

case_dir = SETTINGS.data_dir / st.session_state.case_id
version = (case_dir / "case.json").stat().st_mtime_ns if (case_dir / "case.json").exists() else 0
case = load_case(st.session_state.case_id, str(version))
st.header(case.name)

tab_docs, tab_struct, tab_ask, tab_eval = st.tabs(["1 · Documents", "2 · Structure", "3 · Ask", "4 · Evaluation"])

# --------------------------------------------------------------------------------------------- documents
with tab_docs:
    c1, c2 = st.columns([3, 2])
    with c1:
        ups = st.file_uploader("Upload 2–5 public legal PDFs", type="pdf", accept_multiple_files=True)
        if ups and st.button(f"Add {len(ups)} file(s)"):
            for u in ups:
                case.add_pdf(u.getvalue(), u.name)
            st.success("Added. Now click **Index documents**.")
    with c2:
        url = st.text_input("…or add a PDF by URL", placeholder="https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679")
        if st.button("Download", disabled=not url):
            try:
                case.add_url(url)
                st.success("Downloaded.")
            except Exception as exc:
                st.error(f"Could not download: {exc}")

    files = case.pdf_files()
    st.subheader(f"Files in this case ({len(files)})")
    if not 2 <= len(files) <= 5:
        st.caption("The brief expects a bundle of 2–5 PDFs.")
    for f in files:
        a, b = st.columns([6, 1])
        a.write(f"📄 {f.name} · {f.stat().st_size / 1024:.0f} KB")
        if b.button("Remove", key=f"rm-{f.name}"):
            case.remove_pdf(f.name)
            st.rerun()

    if files and st.button("⚙️ Index documents", type="primary"):
        bar = st.progress(0.0, "Starting…")
        info = case.build_index(progress=lambda p, msg: bar.progress(min(p, 1.0), msg))
        bar.progress(1.0, "Done")
        load_case.clear()
        st.success(f"Indexed {info['n_chunks']} chunks · dense: {info['dense']} · reranker: {info['rerank']}")
        st.rerun()
    if case.meta.get("index"):
        st.caption(f"Index built {case.meta['index']['built']} · {case.meta['index']['n_chunks']} chunks · "
                   f"dense `{case.meta['index']['dense']}` · reranker `{case.meta['index']['rerank']}`")

# --------------------------------------------------------------------------------------------- structure
with tab_struct:
    if not case.documents:
        st.info("Index the documents first.")
    for d in case.documents:
        flagged = [p for p in d.pages if p.scanned or p.anomalous]
        with st.expander(f"📄 {d.name} — “{d.title}” · {len(d.pages)} pages · "
                         f"{sum(p.n_tables for p in d.pages)} tables · {len(flagged)} flagged", expanded=True):
            df = pd.DataFrame([{
                "page": p.number, "chars": p.n_chars, "columns": p.columns, "tables": p.n_tables,
                "images": p.n_images, "scanned": "⚠️ yes" if p.scanned else "",
                "OCR": f"✓ conf {p.ocr_confidence:.2f}" if p.ocr_used else "",
                "AE anomaly z": p.anomaly_score, "anomalous": "⚠️ yes" if p.anomalous else "",
                "headers/footers removed": len(p.removed)} for p in d.pages])
            st.dataframe(df, hide_index=True, width="stretch")
            secs = [c for c in case.chunks if c.doc_id == d.doc_id]
            st.markdown("**Detected sections / clauses**")
            st.dataframe(pd.DataFrame([{"section": c.section, "path": c.heading_path, "pages": c.page_label,
                                        "type": c.kind, "words": len(c.text.split())} for c in secs]),
                         hide_index=True, width="stretch", height=min(35 * len(secs) + 40, 320))
            if flagged:
                st.markdown("**Flagged pages** (scanned → OCR, or unusual layout found by the page autoencoder)")
                cols = st.columns(min(len(flagged), 4))
                for col, p in zip(cols, flagged):
                    reason = "scanned/OCR" if p.scanned else f"AE z={p.anomaly_score}"
                    col.image(case.render_page(d.doc_id, p.number, zoom=0.6), caption=f"p. {p.number} · {reason}")

# --------------------------------------------------------------------------------------------- ask
with tab_ask:
    if not case.chunks:
        st.info("Index the documents first.")
    else:
        st.caption("Try: " + " · ".join(f"`{e}`" for e in EXAMPLES[:2]))
        q = st.text_area("Question", value=st.session_state.get("q", ""), height=80,
                         placeholder="e.g. Does the liability cap in Section 7 exceed the total indemnity obligations in Section 9?")
        ask = st.button("Ask", type="primary", disabled=not q.strip())
        if ask:
            st.session_state.q = q
            with st.spinner("Planning → retrieving → drafting → verifying…"):
                qa = QAPipeline(case, llm=llm, mode=mode, use_verifier=use_verifier)
                st.session_state.setdefault("answers", []).insert(0, qa.ask(q.strip()))

        for n, a in enumerate(st.session_state.get("answers", [])[:5]):
            if n:
                st.divider()
                st.caption("Earlier question")
            st.markdown(f"#### {a.question}")
            st.markdown(badge(a.confidence), unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.1rem;margin:0.6rem 0'>{a.answer}</div>", unsafe_allow_html=True)
            if a.abstained:
                st.warning(f"**Abstained.** {a.explanation}  \nDocuments searched: {', '.join(a.searched)}  \n"
                           f"Best evidence score: {a.evidence_score:.2f}")
            if a.explanation and not a.abstained:
                st.info(f"**Why this answer:** {a.explanation}")
            for c in a.conflicts:
                st.warning(f"⚖️ **Conflict between sources:** {c}")

            if a.claims:
                st.markdown("##### Evidence")
                for i, cl in enumerate(a.claims, 1):
                    st.markdown(f"**Claim {i}.** {cl.text}  " + (f"<span style='color:#1b7f3b'>✔ verified</span>"
                                                                 if cl.supported else ""), unsafe_allow_html=True)
                    for ci in cl.citations:
                        left, right = st.columns([2, 3])
                        with left:
                            st.markdown(f"📌 **{ci.label}**")
                            st.markdown(f"> {ci.quote}")
                            exact = any(r.get("exact") for r in ci.regions)
                            st.caption("🟧 exact phrase box" if exact else "🟦 paragraph box")
                        with right:
                            for p in ci.pages[:2]:
                                st.image(case.render_page(ci.doc_id, p, ci.regions, zoom=1.3),
                                         caption=f"{ci.doc_name} – page {p}", width="stretch")

            with st.expander("🔍 Pipeline details: plan, verifier log, retrieved context, latency"):
                st.markdown("**Search plan:** " + " · ".join(f"`{x}`" for x in a.queries))
                if a.retry_log:
                    st.markdown("**Verifier / retry log**")
                    st.json(a.retry_log, expanded=False)
                if a.context:
                    st.dataframe(pd.DataFrame([{**{k: v for k, v in c.items() if k not in ("text", "chars")},
                                                "ranks": json.dumps(c["ranks"])} for c in a.context]),
                                 hide_index=True, width="stretch")
                m = st.columns(5)
                m[0].metric("Total", f"{a.timings.get('total_s', 0):.2f}s")
                m[1].metric("Retrieve", f"{a.timings.get('retrieve_s', 0):.2f}s")
                m[2].metric("Generate", f"{a.timings.get('generate_s', 0):.2f}s")
                m[3].metric("Prompt tokens", a.tokens["prompt"])
                m[4].metric("Completion tokens", a.tokens["completion"])
                st.caption(f"Backend: {a.backend} · evidence score {a.evidence_score:.2f}")

# --------------------------------------------------------------------------------------------- evaluation
with tab_eval:
    reports = sorted((ROOT / "results").glob("*/eval_report.md"))
    if not reports:
        st.info("Run `make eval` to produce an evaluation report.")
    for r in reports:
        with st.expander(f"Report: {r.parent.name}", expanded=True):
            st.markdown(r.read_text(encoding="utf-8", errors="replace"))
