"""Pipeline tests with a scripted fake LLM, so the planner + verifier loop is tested without a model."""
import json
import re

from lexisync.answer import GEN_SYSTEM, PLAN_SYSTEM, VERIFY_SYSTEM, QAPipeline, detect_amendment_conflicts
from lexisync.llm import LLMResult


class FakeLLM:
    """1st draft hallucinates the cap (EUR 3,000,000); after verifier feedback it answers correctly."""
    name = "fake"

    def __init__(self, insufficient=False):
        self.calls = []
        self.insufficient = insufficient

    def complete(self, system, user, json_mode=True, temperature=0.0):
        self.calls.append(system)
        if system == PLAN_SYSTEM:
            out = {"queries": ["Section 7.2 aggregate liability cap"]}
        elif system == VERIFY_SYSTEM:
            out = {"results": [{"i": i["i"], "supported": True, "reason": "ok"} for i in json.loads(user)]}
        elif system == GEN_SYSTEM:
            cid = re.search(r"\[(C\d+)\] \(document: [^)]*Section 7\.2", user)
            cid = cid.group(1) if cid else "C1"
            if self.insufficient:
                out = {"insufficient": True, "claims": []}
            elif "VERIFIER FEEDBACK" not in user:
                out = {"insufficient": False, "answer": "The cap is EUR 3,000,000.",
                       "claims": [{"text": "The aggregate liability cap is EUR 3,000,000.", "cite": [cid]}],
                       "explanation": "draft 1", "conflicts": []}
            else:
                out = {"insufficient": False, "answer": "The cap is EUR 2,000,000 (Section 7.2).",
                       "claims": [{"text": "The aggregate liability cap is EUR 2,000,000.", "cite": [cid]}],
                       "explanation": "draft 2", "conflicts": []}
        else:
            out = {}
        return LLMResult(json.dumps(out), 100, 20, 0.01)


def test_verifier_rejects_hallucinated_number_then_retries(case):
    qa = QAPipeline(case, llm=FakeLLM())
    a = qa.ask("What is the aggregate liability cap in Section 7.2?")
    events = [e["event"] for e in a.retry_log]
    assert events[0] == "draft-rejected" and "verified" in events
    assert "3000000" in json.dumps(a.retry_log[0]["rejected"]).replace(".0", "").replace(",", "") or \
        "not found" in a.retry_log[0]["rejected"][0]["reason"]
    assert not a.abstained and "2,000,000" in a.answer
    assert a.claims[0].citations[0].section == "Section 7.2"
    assert a.claims[0].citations[0].pages == [4]
    assert a.confidence in ("MEDIUM", "LOW")  # a retry happened, so not HIGH
    assert a.tokens["prompt"] > 0


def test_generator_insufficient_leads_to_abstention(case):
    a = QAPipeline(case, llm=FakeLLM(insufficient=True)).ask("What is the aggregate liability cap in Section 7.2?")
    assert a.abstained and a.confidence == "INSUFFICIENT EVIDENCE"
    assert "02_amendment_no1.pdf" in a.answer  # states which documents were searched


def test_extractive_mode_answers_and_abstains(case):
    qa = QAPipeline(case, llm=None)
    a = qa.ask("Within how many hours must the Processor notify the Controller of a personal data breach?")
    assert not a.abstained and "48" in a.answer
    assert a.claims[0].citations[0].section == "Article 4(1)"
    assert any(r["exact"] for r in a.claims[0].citations[0].regions)  # tight bounding box on the number
    u = qa.ask("What is the governing law of the subcontractor agreement?")
    assert u.abstained


def test_amendment_conflict_detected(case):
    amend = next(c for c in case.chunks if c.section == "Section 2.2" and c.doc_name.startswith("02"))
    orig = next(c for c in case.chunks if c.section == "Section 7.2")
    notes = detect_amendment_conflicts([amend, orig])
    assert notes and "prevails" in notes[0]


class ScriptedLLM:
    """Returns a fixed generator draft; planner/verifier say OK."""
    name = "scripted"

    def __init__(self, make_draft):
        self.make_draft = make_draft

    def complete(self, system, user, json_mode=True, temperature=0.0):
        if system == PLAN_SYSTEM:
            out = {"queries": []}
        elif system == VERIFY_SYSTEM:
            out = {"results": [{"i": i["i"], "supported": True} for i in json.loads(user)]}
        else:
            out = self.make_draft(user)
        return LLMResult(json.dumps(out), 10, 5, 0.01)


def _cid(user, pattern):
    m = re.search(r"\[(C\d+)\] \(document: [^)]*" + pattern, user)
    return m.group(1) if m else "C1"


def test_citation_repair_for_number_from_uncited_passage(case):
    def draft(user):
        c72 = _cid(user, r"Section 7\.2")
        return {"insufficient": False, "answer": "No.",
                "claims": [{"text": "The cap of EUR 2,000,000 is lower than the total indemnities of EUR 3,500,000.",
                            "cite": [c72]}], "explanation": "", "conflicts": []}
    a = QAPipeline(case, llm=ScriptedLLM(draft)).ask(
        "Does the liability cap in Section 7 exceed the total indemnity obligations in Section 9?")
    assert not a.abstained
    assert {"Section 7.2", "Table 9.1"} <= {ci.section for ci in a.claims[0].citations}


def test_claim_about_entity_in_no_document_is_rejected(case):
    def draft(user):
        return {"insufficient": False, "answer": "German law.",
                "claims": [{"text": "The subcontractor agreement is governed by German law.",
                            "cite": [_cid(user, r"Section 12\.1")]}], "explanation": "", "conflicts": []}
    a = QAPipeline(case, llm=ScriptedLLM(draft)).ask("What is the governing law of the subcontractor agreement?")
    assert a.abstained
