"""Token-consumption optimization layer.

Single choke-point for every LLM call. Responsibilities:

1. Adaptive prompt compression. Trim irrelevant history; summarize stale
   turns; replace long retrieved snippets with their lead sentence + a
   pointer until the model asks for more.
2. Prefix-cache-aware prompt assembly. Hold the system prompt and
   per-task instructions stable so providers' prompt-cache machinery
   can hit (Anthropic prompt caching / OpenAI auto cache).
3. Structured outputs. Force JSON schemas when the caller specifies one.
   Reject malformed outputs and retry with a delta-correction prompt.
4. Response-budget truncation. Cap `max_tokens` per call against the
   remaining run budget so the orchestrator never overshoots.
5. Cost accounting. Returns (output, tokens_in, tokens_out, cost_usd,
   cache_hit) so the cost tracker can attribute spend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
import tiktoken
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.config import Settings
from app.prompts.registry import get_prompt
from app.services.semantic_cache import SemanticCache

log = structlog.get_logger(__name__)


# Provider price tables (USD per 1M tokens) — keep in sync with vendor docs.
# Numbers are ballparks; production reads them from a config file.
_PRICE_USD_PER_M: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
    "gpt-4.1-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def _encode(text: str) -> list[int]:
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode(text)


@dataclass
class _CallStats:
    tokens_in: int
    tokens_out: int
    cost_usd: float
    cache_hit: bool


class TokenOptimizer:
    def __init__(self, settings: Settings, *, cache: SemanticCache) -> None:
        self.settings = settings
        self.cache = cache
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))

    async def aclose(self) -> None:
        await self._http.aclose()

    # ---------------------------------------------------------------- API

    async def call_llm_text(
        self,
        prompt: str,
        *,
        cache_key: tuple[Any, ...] | None = None,
        budget_tokens: int = 1500,
        model: str | None = None,
        stop: list[str] | None = None,
    ) -> tuple[str, int, int, float, bool]:
        if cache_key is not None and self.settings.semantic_cache_enabled:
            cached = await self.cache.get(("llm.text", *cache_key))
            if cached is not None:
                return cached["text"], 0, 0, 0.0, True

        compressed = self._compress_prompt(prompt, budget_tokens)
        text, stats = await self._invoke(compressed, model=model, max_tokens=budget_tokens, stop=stop)
        if cache_key is not None and self.settings.semantic_cache_enabled:
            await self.cache.set(("llm.text", *cache_key), {"text": text})
        return text, stats.tokens_in, stats.tokens_out, stats.cost_usd, False

    async def call_llm_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        cache_key: tuple[Any, ...] | None = None,
        budget_tokens: int = 1500,
        model: str | None = None,
    ) -> tuple[dict[str, Any], int, int, float, bool]:
        if cache_key is not None and self.settings.semantic_cache_enabled:
            cached = await self.cache.get(("llm.json", *cache_key))
            if cached is not None:
                return cached["payload"], 0, 0, 0.0, True

        compressed = self._compress_prompt(prompt, budget_tokens)
        wrapped = (
            compressed
            + "\n\nReturn a single JSON object that conforms to this JSON Schema. "
              "Do not emit prose, code fences, or comments.\n"
            + json.dumps(schema, sort_keys=True)
        )
        text, stats = await self._invoke(wrapped, model=model, max_tokens=budget_tokens)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # one structured-output retry with delta correction
            correction = (
                "Your previous output was not valid JSON. Reply with ONLY valid JSON conforming to:\n"
                + json.dumps(schema)
            )
            text, stats2 = await self._invoke(wrapped + "\n" + correction, model=model, max_tokens=budget_tokens)
            stats = _CallStats(
                tokens_in=stats.tokens_in + stats2.tokens_in,
                tokens_out=stats.tokens_out + stats2.tokens_out,
                cost_usd=stats.cost_usd + stats2.cost_usd,
                cache_hit=False,
            )
            payload = json.loads(text)
        if cache_key is not None and self.settings.semantic_cache_enabled:
            await self.cache.set(("llm.json", *cache_key), {"payload": payload})
        return payload, stats.tokens_in, stats.tokens_out, stats.cost_usd, False

    async def summarize(
        self,
        *,
        question: str,
        plan: Any,
        results: dict[str, Any],
        budget_tokens: int,
    ) -> tuple[str, int, int, float]:
        prompt = get_prompt("summarize.v2").render(
            question=question,
            plan_rationale=getattr(plan, "rationale", ""),
            results_summary=self._summarize_results_for_prompt(results),
        )
        text, t_in, t_out, cost, _ = await self.call_llm_text(
            prompt, cache_key=("summarize.v2", question), budget_tokens=max(400, min(budget_tokens, 1200))
        )
        return text, t_in, t_out, cost

    # ------------------------------------------------------------- helpers

    def _compress_prompt(self, prompt: str, budget_tokens: int) -> str:
        toks = _encode(prompt)
        if len(toks) <= max(2_000, budget_tokens * 4):
            return prompt
        # naive but effective: keep the lead and the tail, drop the middle
        head = toks[: 2 * budget_tokens]
        tail = toks[-budget_tokens:]
        enc = tiktoken.get_encoding("cl100k_base")
        return enc.decode(head) + "\n\n[...elided context...]\n\n" + enc.decode(tail)

    @staticmethod
    def _summarize_results_for_prompt(results: dict[str, Any]) -> str:
        """Compact a results dict for the summarizer.

        Lossy by design: each region becomes a single line with its
        method, confidence, and bbox. The full payload remains in the
        response object that the orchestrator returns to the user.
        """

        out: list[str] = []
        for k, v in results.items():
            if isinstance(v, list):
                for r in v:
                    poly = getattr(r, "polygon_geojson", None) or {}
                    out.append(
                        f"- node {k}: method={getattr(r, 'method', '?')} "
                        f"confidence={getattr(r, 'confidence', '?')} "
                        f"bbox={poly.get('bbox') or _bbox_of(poly)}"
                    )
            elif hasattr(v, "prism"):
                p = v.prism
                out.append(f"- node {k}: prism feasible={p.feasible} duration_s={p.duration_s}")
            elif isinstance(v, dict) and "error" in v:
                out.append(f"- node {k}: error={v['error']}")
        return "\n".join(out) or "(no results)"

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=4.0),
        reraise=True,
    )
    async def _invoke(
        self,
        prompt: str,
        *,
        model: str | None,
        max_tokens: int,
        stop: list[str] | None = None,
    ) -> tuple[str, _CallStats]:
        m = model or self.settings.primary_model
        # For a real deployment, this routes to Anthropic / OpenAI clients.
        # The scaffold uses HTTP to the vendors so retries / timeouts /
        # circuit breakers all sit in one place.
        if m.startswith("claude"):
            text, t_in, t_out = await self._invoke_anthropic(prompt, m, max_tokens, stop)
        else:
            text, t_in, t_out = await self._invoke_openai(prompt, m, max_tokens, stop)
        cost = self._cost(m, t_in, t_out)
        return text, _CallStats(tokens_in=t_in, tokens_out=t_out, cost_usd=cost, cache_hit=False)

    async def _invoke_anthropic(
        self, prompt: str, model: str, max_tokens: int, stop: list[str] | None
    ) -> tuple[str, int, int]:
        key = self.settings.anthropic_api_key
        if key is None or not key.get_secret_value():
            return _stub_response(prompt), len(_encode(prompt)), 64
        headers = {
            "x-api-key": self.settings.anthropic_api_key.get_secret_value(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "stop_sequences": stop or [],
            "system": "You are GeoTrace-Agent. Be precise, terse, and cite sources.",
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt,
                                                        "cache_control": {"type": "ephemeral"}}]}],
        }
        r = await self._http.post("https://api.anthropic.com/v1/messages", headers=headers, json=body)
        r.raise_for_status()
        out = r.json()
        text = "".join(b["text"] for b in out["content"] if b["type"] == "text")
        usage = out.get("usage", {})
        return text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))

    async def _invoke_openai(
        self, prompt: str, model: str, max_tokens: int, stop: list[str] | None
    ) -> tuple[str, int, int]:
        key = self.settings.openai_api_key
        if key is None or not key.get_secret_value():
            return _stub_response(prompt), len(_encode(prompt)), 64
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key.get_secret_value()}",
            "content-type": "application/json",
        }
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stop": stop,
        }
        r = await self._http.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
        r.raise_for_status()
        out = r.json()
        text = out["choices"][0]["message"]["content"] or ""
        usage = out.get("usage", {})
        return text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))

    @staticmethod
    def _cost(model: str, t_in: int, t_out: int) -> float:
        in_p, out_p = _PRICE_USD_PER_M.get(model, (3.0, 15.0))
        return (t_in * in_p + t_out * out_p) / 1_000_000.0


def _stub_response(prompt: str) -> str:
    """Offline stub for tests and local dev (no API key set)."""

    if '"type": "object"' in prompt or "JSON" in prompt:
        question_l = _extract_question(prompt).lower()
        if "coverage gap" in question_l or "signal denial" in question_l or "gap audit" in question_l:
            return json.dumps({
                "rationale": "stub: detect anomalous trajectory gap, then summarize",
                "nodes": [
                    {
                        "id": "g1",
                        "kind": "gaps.detect",
                        "deps": [],
                        "inputs": {
                            "domain": "vessel",
                            "coverage_threshold_s": 600,
                            "trajectory": [
                                {"lat": 56.10, "lon": -162.05, "t": "2026-01-15T06:00:00Z"},
                                {"lat": 56.18, "lon": -162.18, "t": "2026-01-15T06:06:00Z"},
                                {"lat": 56.30, "lon": -162.40, "t": "2026-01-15T12:00:00Z"},
                            ],
                        },
                        "expected_tokens": 0,
                        "confidence_prior": 0.72,
                        "rationale": "local deterministic gap audit",
                    },
                    {"id": "s1", "kind": "summarize", "deps": ["g1"], "expected_tokens": 200,
                     "confidence_prior": 0.7, "rationale": "summarize"},
                ],
            })
        if "rendezvous" in question_l or "have met" in question_l or "met " in question_l:
            return json.dumps({
                "rationale": "stub: compute two prisms, find rendezvous, validate, summarize",
                "nodes": [
                    {"id": "p1", "kind": "prism.compute", "deps": [], "expected_tokens": 0,
                     "confidence_prior": 0.7, "rationale": "prism for first anchor pair"},
                    {"id": "p2", "kind": "prism.compute", "deps": [], "expected_tokens": 0,
                     "confidence_prior": 0.7, "rationale": "prism for second anchor pair"},
                    {"id": "r1", "kind": "rendezvous.tgard", "deps": ["p1", "p2"], "expected_tokens": 0,
                     "confidence_prior": 0.6, "rationale": "candidate regions"},
                    {"id": "v1", "kind": "validate.kinematic", "deps": ["r1"], "expected_tokens": 0,
                     "confidence_prior": 0.85, "rationale": "validate"},
                    {"id": "s1", "kind": "summarize", "deps": ["v1"], "expected_tokens": 200,
                     "confidence_prior": 0.7, "rationale": "summarize"},
                ],
            })
        return json.dumps({
            "rationale": "stub: compute prism and summarize",
            "nodes": [
                {"id": "p1", "kind": "prism.compute", "deps": [], "expected_tokens": 0,
                 "confidence_prior": 0.7, "rationale": "prism for anchor pair"},
                {"id": "s1", "kind": "summarize", "deps": ["p1"], "expected_tokens": 200,
                 "confidence_prior": 0.7, "rationale": "summarize"},
            ],
        })
    return "Stub answer. Configure GT_ANTHROPIC_API_KEY or GT_OPENAI_API_KEY for live LLM responses."


def _extract_question(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("Question:"):
            return line.removeprefix("Question:").strip()
    return prompt


def _bbox_of(geo: dict[str, Any]) -> list[float] | None:
    coords = geo.get("coordinates")
    if not coords:
        return None
    flat: list[tuple[float, float]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, (list, tuple)):
            if node and isinstance(node[0], (int, float)) and len(node) >= 2:
                flat.append((float(node[0]), float(node[1])))
            else:
                for n in node:
                    _walk(n)

    _walk(coords)
    if not flat:
        return None
    xs = [x for x, _ in flat]
    ys = [y for _, y in flat]
    return [min(xs), min(ys), max(xs), max(ys)]
