"""Question answering pipeline: plan -> retrieve -> generate (cited) -> verify -> retry | abstain.

Planner + verifier agent loop (extended feature #5):
  1. PLAN     – decompose the question into <= 3 search queries (LLM, or heuristic for clause references).
  2. RETRIEVE – hybrid BM25 + dense + cross-encoder for each query; merged context C1..Cn.
  3. GATE     – if the best evidence score is below a threshold -> abstain before generating.
  4. DRAFT    – the LLM writes JSON {answer, claims[{text, cite}], explanation, conflicts, insufficient}.
  5. VERIFY   – every claim must cite a valid passage; its numbers must occur in (or be derivable from) the
                cited text; and an LLM judge checks entailment. A rejected draft is logged, the unsupported
                claims become new search queries and the loop repeats (step cap = max_verify_steps).
  6. FINAL    – unsupported claims are removed; no supported claim left -> abstain. Confidence badge.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field

from .chunking import Chunk
from .config import SETTINGS
from .grounding import best_regions, numbers, numeric_support, token_overlap
from .llm import get_llm
from .retrieval import Hit, content_terms, tokenize

log = logging.getLogger(__name__)

ABSTAIN_MSG = "The documents do not contain enough evidence to answer this question."

GEN_SYSTEM = """You are LexiSync, a careful legal document analyst.
Rules:
- Answer ONLY from the numbered context passages. Never use outside knowledge.
- Every factual claim must cite the passage ids that support it, e.g. ["C2"]. Quote numbers exactly as written.
- If the passages do not contain the evidence needed, set "insufficient": true and leave claims empty.
  Partial evidence for a different document, party or agreement than the one asked about is NOT evidence.
- If passages conflict (e.g. an amendment changes a value), state both values with citations and say which one
  prevails and why (later amendment, explicit precedence clause, more specific provision) in "conflicts".
- Write the answer in plain language for a lawyer or compliance officer; start with a direct answer (Yes/No/value).
Return JSON only:
{"insufficient": false,
 "answer": "direct plain-language answer",
 "claims": [{"text": "one atomic factual statement", "cite": ["C1"]}],
 "explanation": "why the cited evidence supports the answer and how passages were combined",
 "conflicts": ["..."]}"""

PLAN_SYSTEM = """You plan searches over a bundle of legal documents.
Split the user question into at most 3 short, standalone search queries that together retrieve all the
evidence needed (one per fact / clause / document). Keep clause numbers (e.g. "Section 7.2", "Article 83(5)")
and defined terms exactly. If one query suffices, return one. JSON only: {"queries": ["..."]}"""

VERIFY_SYSTEM = """You are a strict verifier. For each claim decide whether the cited evidence text fully
supports it (entailment). Numbers, parties, documents and conditions must match.
A comparison or simple arithmetic on numbers that appear in the evidence (e.g. "2,000,000 is lower than 3,500,000")
IS supported. A claim about a different document, agreement or party than the evidence is NOT supported. JSON only:
{"results": [{"i": 0, "supported": true, "reason": "short"}]}"""

NO_RETRIEVAL_SYSTEM = """You are a legal analyst. Answer the question from your own knowledge.
If you do not know, set "insufficient": true. JSON only:
{"insufficient": false, "answer": "...", "claims": [{"text": "...", "cite": []}], "explanation": "...", "conflicts": []}"""


# ------------------------------------------------------------------------------------------ data model
@dataclass
class Citation:
    cid: str
    chunk_id: str
    doc_id: str
    doc_name: str
    section: str
    pages: list[int]
    quote: str
    regions: list[dict] = field(default_factory=list)

    @property
    def label(self) -> str:
        p = self.pages
        pg = f"p. {p[0]}" if len(p) == 1 else f"pp. {p[0]}–{p[-1]}"
        return f"{self.doc_name} · {pg} · {self.section}"


@dataclass
class Claim:
    text: str
    citations: list[Citation]
    supported: bool | None = None
    reason: str = ""


@dataclass
class Answer:
    question: str
    answer: str
    claims: list[Claim]
    confidence: str
    abstained: bool
    explanation: str = ""
    conflicts: list[str] = field(default_factory=list)
    searched: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    retry_log: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=lambda: {"prompt": 0, "completion": 0})
    backend: str = ""
    evidence_score: float = 0.0
    context: list[dict] = field(default_factory=list)

    @property
    def cited_pages(self) -> list[tuple[str, int]]:
        return sorted({(c.doc_name, p) for cl in self.claims for c in cl.citations for p in c.pages})

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------------ pipeline
class QAPipeline:
    def __init__(self, case, llm="auto", mode: str = "hybrid_rerank", use_verifier: bool = True,
                 context_k: int | None = None, rerank_depth: int | None = None, use_planner: bool = True):
        self.case = case
        self.retriever = case.retriever
        self.llm = get_llm() if llm == "auto" else llm
        self.mode = mode
        self.use_verifier = use_verifier
        self.use_planner = use_planner
        self.context_k = context_k or SETTINGS.context_k
        self.rerank_depth = rerank_depth
        self.backend = self.llm.name if self.llm else "extractive (no LLM)"

    # ---------------------------------------------------------------- public
    def ask(self, question: str, no_retrieval: bool = False) -> Answer:
        t_start = time.perf_counter()
        ans = Answer(question=question, answer="", claims=[], confidence="", abstained=False, backend=self.backend,
                     searched=[d.name for d in self.case.documents])
        if no_retrieval:
            return self._no_retrieval(ans, t_start)

        # 1. plan
        t0 = time.perf_counter()
        ans.queries = self._plan(question, ans)
        ans.timings["plan_s"] = time.perf_counter() - t0

        # 2. retrieve
        t0 = time.perf_counter()
        hits = self._retrieve_multi(ans.queries)
        ans.timings["retrieve_s"] = time.perf_counter() - t0
        ans.evidence_score = round(max((h.evidence for h in hits), default=0.0), 3)
        missing = self.retriever.missing_terms(question)

        # 3. evidence gate
        gate_reason = None
        thr = SETTINGS.abstain_threshold if self.retriever.cross is not None else SETTINGS.abstain_threshold_lexical
        if ans.evidence_score < thr:
            gate_reason = f"best evidence score {ans.evidence_score:.2f} < threshold {thr}"
        elif (unknown := self.retriever.unknown_document(question)):
            gate_reason = f"the question asks about the '{unknown}', which is not among the documents in this case"
        elif self.llm is None and missing:
            gate_reason = f"key term(s) {missing} occur in none of the documents"
        if gate_reason:
            ans.retry_log.append({"step": 0, "event": "abstain-gate", "detail": gate_reason})
            ans.context = [self._ctx_dict(f"C{i+1}", h) for i, h in enumerate(hits)]
            return self._abstain(ans, t_start, gate_reason)

        # 4–5. draft + verify loop
        feedback = ""
        draft = {}
        for step in range(1, SETTINGS.max_verify_steps + 1):
            ctx = {f"C{i+1}": h for i, h in enumerate(hits)}
            t0 = time.perf_counter()
            draft = self._draft(question, ctx, feedback, ans)
            ans.timings["generate_s"] = ans.timings.get("generate_s", 0) + time.perf_counter() - t0
            if draft.get("insufficient") or not draft.get("claims"):
                ans.retry_log.append({"step": step, "event": "draft-insufficient",
                                      "detail": "generator reported insufficient evidence"})
                ans.context = [self._ctx_dict(k, h) for k, h in ctx.items()]
                return self._abstain(ans, t_start, "generator found no supporting evidence")
            claims = self._build_claims(draft, ctx)
            if not self.use_verifier:
                break
            t0 = time.perf_counter()
            self._verify(claims, ans, ctx)
            ans.timings["verify_s"] = ans.timings.get("verify_s", 0) + time.perf_counter() - t0
            bad = [c for c in claims if not c.supported]
            ans.retry_log.append({"step": step, "event": "verified" if not bad else "draft-rejected",
                                  "detail": f"{len(claims) - len(bad)}/{len(claims)} claims supported",
                                  "rejected": [{"claim": c.text, "reason": c.reason} for c in bad]})
            if not bad:
                break
            if step == SETTINGS.max_verify_steps:
                ans.retry_log.append({"step": step, "event": "step-cap", "detail": "max steps reached; "
                                      "dropping unsupported claims"})
                break
            # reformulate: unsupported claims become new queries, context is widened
            new_q = [re.sub(r"\[C\d+\]", "", c.text)[:200] for c in bad][:2]
            ans.queries += new_q
            more = self._retrieve_multi(new_q)
            seen = {h.chunk.chunk_id for h in hits}
            hits = hits + [h for h in more if h.chunk.chunk_id not in seen][:3]
            feedback = ("A verifier REJECTED these claims of your previous draft because the cited passages do not "
                        "support them: " + json.dumps([{"claim": c.text, "reason": c.reason} for c in bad]) +
                        ". Rewrite the answer using only supported statements; if the evidence is missing, set "
                        "insufficient=true.")

        ans.context = [self._ctx_dict(f"C{i+1}", h) for i, h in enumerate(hits)]
        supported = [c for c in claims if c.supported is not False]
        dropped = len(claims) - len(supported)
        if not supported:
            return self._abstain(ans, t_start, "no claim survived verification")
        ans.claims = supported
        ans.answer = draft.get("answer", "").strip() or " ".join(c.text for c in supported)
        if dropped and self.llm is not None:
            ans.answer += f" (Note: {dropped} unsupported statement(s) were removed by the verifier.)"
        ans.explanation = draft.get("explanation", "")
        ans.conflicts = list(dict.fromkeys((draft.get("conflicts") or []) + detect_amendment_conflicts(
            [h.chunk for h in hits if any(c.chunk_id == h.chunk.chunk_id for cl in supported for c in cl.citations)]
            or [h.chunk for h in hits[:4]])))
        ans.confidence = self._confidence(ans, dropped)
        ans.timings["total_s"] = time.perf_counter() - t_start
        return ans

    # ---------------------------------------------------------------- planning & retrieval
    def _plan(self, question: str, ans: Answer) -> list[str]:
        queries = [question]
        if not self.use_planner:
            return queries
        if self.llm is not None:
            try:
                r = self.llm.complete(PLAN_SYSTEM, question)
                self._count(ans, r)
                qs = [q for q in r.json().get("queries", []) if isinstance(q, str) and q.strip()]
                queries += [q for q in qs[:3] if q.strip().lower() != question.strip().lower()]
            except Exception as exc:
                log.warning("planner failed: %s", exc)
        else:
            queries += heuristic_plan(question)
        return list(dict.fromkeys(queries))

    def _retrieve_multi(self, queries: list[str]) -> list[Hit]:
        """Retrieve for each query and interleave results (round robin) so every sub-question is represented."""
        per_q = [self.retriever.retrieve(q, k=self.context_k, mode=self.mode, rerank_depth=self.rerank_depth)
                 for q in queries]
        merged, seen = [], set()
        budget = self.context_k + 2 * (len(queries) - 1)
        for rank in range(self.context_k):
            for hits in per_q:
                if rank < len(hits) and hits[rank].chunk.chunk_id not in seen:
                    seen.add(hits[rank].chunk.chunk_id)
                    merged.append(hits[rank])
        return merged[:min(budget, 10)]

    # ---------------------------------------------------------------- generation
    def _draft(self, question: str, ctx: dict[str, Hit], feedback: str, ans: Answer) -> dict:
        if self.llm is None:
            return extractive_draft(question, ctx, self.retriever)
        passages = "\n\n".join(
            f"[{cid}] (document: {h.chunk.doc_name} | {h.chunk.section} | {h.chunk.page_label}"
            f"{' | OCR text' if h.chunk.kind == 'ocr' else ''})\n{h.chunk.text}" for cid, h in ctx.items())
        user = f"Question: {question}\n\nContext passages:\n{passages}"
        missing = self.retriever.missing_terms(question)
        if missing:
            user += (f"\n\nNOTE: the question terms {missing} occur in NONE of the documents. If the question is about "
                     f"something the documents do not cover, set insufficient=true.")
        if feedback:
            user += f"\n\nVERIFIER FEEDBACK: {feedback}"
        try:
            r = self.llm.complete(GEN_SYSTEM, user)
            self._count(ans, r)
            return r.json()
        except Exception as exc:
            log.error("generation failed: %s", exc)
            ans.retry_log.append({"step": -1, "event": "llm-error", "detail": str(exc)[:200]})
            return extractive_draft(question, ctx, self.retriever)

    def _build_claims(self, draft: dict, ctx: dict[str, Hit]) -> list[Claim]:
        """Robust to different LLM output styles: claims as dicts or plain strings, cite ids as
        "C1", "[C1]", 1, "1", or only written inline in the text."""
        raw = draft.get("claims") or []
        if isinstance(raw, (str, dict)):
            raw = [raw]
        if not raw and draft.get("answer"):  # model put citations only into the answer text
            raw = [s for s in re.split(r"(?<=[.;])\s+", str(draft["answer"])) if re.search(r"C\d+", s)]
        claims = []
        for c in raw:
            if isinstance(c, str):
                c = {"text": c}
            if not isinstance(c, dict):
                continue
            text = str(c.get("text") or c.get("claim") or c.get("statement") or "").strip()
            if not text:
                continue
            ids = (c.get("cite") or c.get("citations") or c.get("citation") or c.get("sources")
                   or c.get("source") or [])
            if not isinstance(ids, list):
                ids = [ids]
            found = re.findall(r"C\d+", text)
            for i in ids:
                for part in re.split(r"[,\s]+", str(i)):
                    if re.fullmatch(r"\[?C?\d+\]?", part.strip()):
                        found.append("C" + re.search(r"\d+", part).group(0))
            cits = []
            for cid in dict.fromkeys(found):
                h = ctx.get(cid)
                if not h:
                    continue
                regions = best_regions(text, h.chunk, lambda p, d=h.chunk.doc_id: self.case.page(d, p))
                cits.append(Citation(cid, h.chunk.chunk_id, h.chunk.doc_id, h.chunk.doc_name, h.chunk.section,
                                     sorted({r["page"] for r in regions}) or h.chunk.pages,
                                     _quote(text, h.chunk), regions))
            claims.append(Claim(re.sub(r"\s*\[?C\d+(?:,\s*C\d+)*\]?", "", text).strip(), cits))
        return claims

    # ---------------------------------------------------------------- verification
    def _make_citation(self, cid: str, h: Hit, text: str) -> Citation:
        regions = best_regions(text, h.chunk, lambda p, d=h.chunk.doc_id: self.case.page(d, p))
        return Citation(cid, h.chunk.chunk_id, h.chunk.doc_id, h.chunk.doc_name, h.chunk.section,
                        sorted({r["page"] for r in regions}) or h.chunk.pages, _quote(text, h.chunk), regions)

    def _verify(self, claims: list[Claim], ans: Answer, ctx: dict[str, Hit] | None = None) -> None:
        q_missing = set(self.retriever.missing_terms(ans.question))
        for c in claims:
            if not c.citations:
                c.supported, c.reason = False, "no valid citation"
                continue
            # a claim about something that occurs in no document (e.g. 'subcontractor') cannot be grounded
            off_topic = sorted(q_missing & set(content_terms(c.text)))
            if off_topic:
                c.supported, c.reason = False, f"claim is about {off_topic}, which occurs in none of the documents"
                continue
            evidence = "\n".join(self._chunk(ci.chunk_id).text for ci in c.citations)
            ok, missing = numeric_support(c.text, evidence)
            if not ok and ctx:
                # citation repair: the number may be in another retrieved passage the model forgot to cite
                cited = {ci.cid for ci in c.citations}
                for cid, h in ctx.items():
                    if cid in cited:
                        continue
                    nums = numbers(h.chunk.text)
                    if any(any(abs(m - n) <= 0.005 * max(abs(n), 1) for n in nums) for m in missing):
                        c.citations.append(self._make_citation(cid, h, c.text))
                        cited.add(cid)
                evidence = "\n".join(self._chunk(ci.chunk_id).text for ci in c.citations)
                ok, missing = numeric_support(c.text, evidence)
            if not ok:
                c.supported, c.reason = False, f"number(s) {missing} not found in cited text"
            elif token_overlap(c.text, evidence) < 0.25:
                c.supported, c.reason = False, "cited text shares too little content with the claim"
            else:
                c.supported, c.reason = True, "numbers and key terms found in cited text"
        to_judge = [c for c in claims if c.supported]
        if self.llm is None or not to_judge:
            return
        payload = [{"i": i, "claim": c.text, "evidence": "\n".join(self._chunk(ci.chunk_id).text
                                                                      for ci in c.citations)[:3000]}
                   for i, c in enumerate(to_judge)]
        try:
            r = self.llm.complete(VERIFY_SYSTEM, json.dumps(payload, ensure_ascii=False))
            self._count(ans, r)
            for res in r.json().get("results", []):
                i = res.get("i")
                if isinstance(i, int) and 0 <= i < len(to_judge) and res.get("supported") is False:
                    to_judge[i].supported, to_judge[i].reason = False, "LLM judge: " + str(res.get("reason", ""))
        except Exception as exc:
            log.warning("LLM verifier failed (%s); keeping deterministic checks", exc)

    # ---------------------------------------------------------------- helpers
    def _chunk(self, chunk_id: str) -> Chunk:
        return next(c for c in self.retriever.chunks if c.chunk_id == chunk_id)

    def _confidence(self, ans: Answer, dropped: int) -> str:
        retried = any(e["event"] == "draft-rejected" for e in ans.retry_log)
        ocr = any(self._chunk(ci.chunk_id).kind == "ocr" for cl in ans.claims for ci in cl.citations)
        anomalous = any(self.case.doc(ci.doc_id).pages[p - 1].anomalous
                        for cl in ans.claims for ci in cl.citations for p in ci.pages)
        if dropped or ans.evidence_score < (SETTINGS.abstain_threshold + SETTINGS.high_conf_threshold) / 2:
            return "LOW"
        if retried or ans.conflicts or ocr or anomalous or ans.evidence_score < SETTINGS.high_conf_threshold:
            return "MEDIUM"
        return "HIGH"

    def _abstain(self, ans: Answer, t_start: float, reason: str) -> Answer:
        ans.abstained = True
        ans.confidence = "INSUFFICIENT EVIDENCE"
        ans.answer = f"{ABSTAIN_MSG} Searched: {', '.join(ans.searched)}."
        ans.explanation = f"Abstained: {reason}."
        ans.timings["total_s"] = time.perf_counter() - t_start
        return ans

    def _no_retrieval(self, ans: Answer, t_start: float) -> Answer:
        if self.llm is None:
            raise RuntimeError("no-retrieval baseline needs an LLM backend")
        r = self.llm.complete(NO_RETRIEVAL_SYSTEM, ans.question)
        self._count(ans, r)
        d = r.json()
        if d.get("insufficient"):
            return self._abstain(ans, t_start, "model said it does not know")
        ans.answer = d.get("answer", "")
        ans.claims = [Claim(c.get("text", ""), []) for c in d.get("claims", []) if isinstance(c, dict)]
        ans.confidence = "UNGROUNDED"
        ans.timings["total_s"] = time.perf_counter() - t_start
        return ans

    @staticmethod
    def _count(ans: Answer, r) -> None:
        ans.tokens["prompt"] += r.prompt_tokens
        ans.tokens["completion"] += r.completion_tokens

    @staticmethod
    def _ctx_dict(cid: str, h: Hit) -> dict:
        return {"cid": cid, "doc": h.chunk.doc_name, "section": h.chunk.section, "pages": h.chunk.pages,
                "score": round(h.score, 3), "evidence": round(h.evidence, 3), "ranks": h.ranks,
                "chars": len(h.chunk.text), "text": h.chunk.text[:600]}


# ------------------------------------------------------------------------------------------ heuristics
REF_PATTERN = re.compile(r"\b(Section|Clause|Article|Table|Schedule|Annex)\s+(\d+[a-z]?(?:\.\d+)*(?:\(\d+\))?|[A-Z])\b")


def heuristic_plan(question: str) -> list[str]:
    """Without an LLM: one sub-query per referenced clause/table (the words leading up to the reference,
    e.g. 'liability cap in Section 7' / 'total indemnity obligations in Section 9'), plus 'X and Y' splits."""
    subs, start = [], 0
    for m in REF_PATTERN.finditer(question):
        seg = question[start:m.end()]
        seg = re.sub(r"^\W*(?:does|do|did|is|are|what|which|how|the|exceed|exceeds|than|compare|with|and)\b\s*", "",
                     seg.strip(), flags=re.I)
        subs.append(seg.strip(" ,?") or m.group(0))
        start = m.end()
    parts = re.split(r"\b(?:and|while|whereas|compared (?:to|with)|versus|vs\.?)\b", question)
    if len(parts) > 1:
        subs += [p.strip(" ,?") for p in parts if len(content_terms(p)) >= 2]
    return [s for s in dict.fromkeys(subs) if s.lower() != question.lower()][:3]


def _sentences(chunk: Chunk) -> list[str]:
    if chunk.kind == "table":
        rows = [r for r in chunk.text.splitlines() if r.startswith("|") and "---" not in r]
        header = rows[0] if rows else ""
        return [f"{chunk.section}: {header} {r}" for r in rows[1:]]
    text = re.sub(r"\s+", " ", chunk.text)
    return [s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z0-9(“\"\[])", text) if len(s.split()) >= 4]


def extractive_draft(question: str, ctx: dict[str, Hit], retriever) -> dict:
    """No-LLM fallback: select the best supporting sentence per (sub-)question from the retrieved passages."""
    queries = [question] + heuristic_plan(question)
    q_terms = set(content_terms(question))
    claims, used = [], set()
    wants_amount = bool(re.search(r"\b(exceed|total|amount|cap|fee|how much|maximum|minimum|limit|sum)\b", question, re.I))
    for q in queries:
        refs = [f"{k} {n}" for k, n in REF_PATTERN.findall(q)]
        best = None
        for rank, (cid, h) in enumerate(ctx.items()):
            if refs and not any(re.search(rf"\b{re.escape(r)}\b", h.chunk.heading_path) for r in refs):
                continue  # the sub-question names a clause: only look inside that clause
            for s in _sentences(h.chunk):
                cov = retriever.lexical_evidence(q, s + " " + h.chunk.heading_path)
                score = cov + 0.15 * h.evidence - 0.01 * rank + (0.25 if wants_amount and numbers(s) else 0)
                if s not in used and (best is None or score > best[0]):
                    best = (score, cov, s, cid)
        terms = content_terms(q)
        rare = max(terms, key=retriever.idf) if terms and not refs else None
        threshold = 0.3 if refs else 0.5
        if best and best[1] >= threshold and (rare is None or retriever.lexical_evidence(rare, best[2] + " " + ctx[best[3]].chunk.heading_path) > 0):
            used.add(best[2])
            claims.append({"text": best[2], "cite": [best[3]]})
    if not claims:
        return {"insufficient": True, "claims": []}
    covered = set().union(*(set(tokenize(c["text"])) for c in claims))
    if q_terms and len(q_terms & covered) / len(q_terms) < 0.4:
        return {"insufficient": True, "claims": []}
    answer = " ".join(f"{c['text']} [{c['cite'][0]}]" for c in claims)
    return {"insufficient": False, "answer": answer, "claims": claims,
            "explanation": "Extractive mode: the sentences with the highest query-term coverage were selected "
                           "from the top-ranked passages.", "conflicts": []}


AMEND_RE = re.compile(r"(Section|Clause|Article)\s+(\d+(?:\.\d+)*)\s+of the (?:Agreement|[A-Z][\w ]+?)\s+is amended", re.I)


def detect_amendment_conflicts(chunks: list[Chunk]) -> list[str]:
    """Deterministic cross-document conflict note: a passage that amends a clause which is also in context."""
    notes = []
    for c in chunks:
        for kind, num in AMEND_RE.findall(c.text):
            target = f"Section {num}" if kind.lower() != "article" else f"Article {num}"
            for o in chunks:
                if o.doc_id != c.doc_id and o.section.startswith(target):
                    notes.append(f"{c.doc_name} ({c.section}, {c.page_label}) amends {o.doc_name} {o.section} "
                                 f"({o.page_label}); the amendment is later in time and prevails where they differ.")
    return list(dict.fromkeys(notes))


def _quote(claim: str, chunk: Chunk, max_len: int = 320) -> str:
    sents = _sentences(chunk) or [chunk.text]
    best = max(sents, key=lambda s: token_overlap(claim, s))
    return best[:max_len]
