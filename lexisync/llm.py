"""LLM backends. Local Ollama is the default (brief: "an LLM (local)"); any OpenAI-compatible endpoint
(Groq, OpenRouter, OCI, vLLM, LM Studio) works as a fallback. If no LLM is reachable the pipeline uses
the extractive answerer, so the app always runs."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

import requests

from .config import SETTINGS

log = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float

    def json(self) -> dict:
        return parse_json(self.text)


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    log.warning("LLM returned non-JSON output: %s", text[:200])
    return {}


class OllamaLLM:
    def __init__(self, model: str | None = None, url: str | None = None):
        self.model = model or SETTINGS.ollama_model
        self.url = (url or SETTINGS.ollama_url).rstrip("/")
        self.name = f"ollama:{self.model}"

    def available(self) -> bool:
        try:
            tags = requests.get(f"{self.url}/api/tags", timeout=3).json()
            names = {m["name"] for m in tags.get("models", [])}
            ok = self.model in names or f"{self.model}:latest" in names
            if not ok:
                log.warning("Ollama is running but model '%s' is not pulled (ollama pull %s)", self.model, self.model)
            return ok
        except Exception:
            return False

    def complete(self, system: str, user: str, json_mode: bool = True, temperature: float = 0.0) -> LLMResult:
        t0 = time.perf_counter()
        body = {"model": self.model, "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": temperature, "seed": SETTINGS.seed, "num_ctx": 8192}}
        if json_mode:
            body["format"] = "json"
        r = requests.post(f"{self.url}/api/chat", json=body, timeout=SETTINGS.llm_timeout)
        r.raise_for_status()
        d = r.json()
        return LLMResult(d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0),
                         time.perf_counter() - t0)


class OpenAICompatLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        self.model = model or SETTINGS.openai_model
        self.base_url = (base_url or SETTINGS.openai_base_url).rstrip("/")
        self.api_key = api_key or SETTINGS.openai_api_key
        self.name = f"api:{self.model}"

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, json_mode: bool = True, temperature: float = 0.0) -> LLMResult:
        t0 = time.perf_counter()
        body = {"model": self.model, "temperature": temperature, "seed": SETTINGS.seed,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        for attempt in range(3):
            r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=SETTINGS.llm_timeout,
                              headers={"Authorization": f"Bearer {self.api_key}"})
            if r.status_code == 429 and attempt < 2:      # free tiers rate-limit: back off and retry
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        d = r.json()
        u = d.get("usage", {})
        return LLMResult(d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0),
                         u.get("completion_tokens", 0), time.perf_counter() - t0)


def get_llm(backend: str | None = None):
    """Return an LLM object, or None for the extractive (no-LLM) mode."""
    backend = backend or SETTINGS.llm_backend
    if backend == "extractive":
        return None
    candidates = [OllamaLLM(), OpenAICompatLLM()] if backend == "ollama" else [OpenAICompatLLM(), OllamaLLM()]
    for llm in candidates:
        if llm.available():
            return llm
    log.warning("No LLM backend reachable -> extractive mode")
    return None
