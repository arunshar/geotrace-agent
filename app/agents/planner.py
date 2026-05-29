"""PlannerAgent. Decomposes a question into a typed PlanGraph (DAG).

The planner's chain-of-thought is structured: rather than emitting free-form
prose, it returns a small set of typed nodes whose schemas the orchestrator
can statically validate. This pattern (a) makes traces auditable, (b) lets the
runtime apply tool-call optimization without parsing free-form text, and
(c) bounds the worst-case token consumption per plan.
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import ValidationError

from app.config import Settings
from app.errors import GuardrailTripped, PlanInfeasible
from app.models import (
    AnchorPair,
    ConversationState,
    PlanGraph,
    PlanNode,
    PlanNodeKind,
    QueryIn,
)
from app.prompts.registry import get_prompt
from app.services.token_optimizer import TokenOptimizer

log = structlog.get_logger(__name__)


class PlannerAgent:
    """Decomposes a question into a `PlanGraph` of typed nodes."""

    def __init__(self, settings: Settings, token_opt: TokenOptimizer) -> None:
        self.settings = settings
        self.token_opt = token_opt
        self.last_tokens_in: int = 0
        self.last_tokens_out: int = 0
        self.last_cost_usd: float = 0.0
        self.last_cache_hit: bool = False

    async def plan(self, q: QueryIn, convo: ConversationState) -> PlanGraph:
        prompt = get_prompt("planner.v3").render(
            question=q.question,
            domain=q.domain,
            anchors=[a.model_dump() for a in (q.anchors or [])],
            budget=q.budget.model_dump(),
            history=convo.history[-4:],
        )
        out, t_in, t_out, cost, cache_hit = await self.token_opt.call_llm_json(
            prompt=prompt,
            schema=_PLAN_JSON_SCHEMA,
            cache_key=("planner.v3", q.question, q.domain),
            budget_tokens=2_500,
        )
        self.last_tokens_in, self.last_tokens_out = t_in, t_out
        self.last_cost_usd = cost
        self.last_cache_hit = cache_hit
        try:
            return _coerce_plan(out, q)
        except (ValidationError, ValueError) as exc:
            raise PlanInfeasible(f"planner emitted invalid plan: {exc}") from exc


# ------------------------------------------------------------ helpers


_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["nodes", "rationale"],
    "properties": {
        "rationale": {"type": "string", "minLength": 4, "maxLength": 1200},
        "nodes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 16,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "kind"],
                "properties": {
                    "id": {"type": "string", "minLength": 1, "maxLength": 16},
                    "kind": {
                        "enum": [
                            "prism.compute",
                            "gaps.detect",
                            "rendezvous.tgard",
                            "rendezvous.dc_tgard",
                            "validate.kinematic",
                            "retrieve.semantic",
                            "summarize",
                        ],
                    },
                    "deps": {"type": "array", "items": {"type": "string"}},
                    "inputs": {"type": "object"},
                    "expected_tokens": {"type": "integer", "minimum": 0, "maximum": 30_000},
                    "confidence_prior": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}


def _coerce_plan(payload: dict[str, Any], q: QueryIn) -> PlanGraph:
    raw_nodes = payload["nodes"]
    nodes: list[PlanNode] = []
    prism_idx = 0
    for n in raw_nodes:
        kind = PlanNodeKind(n["kind"])
        # `prism.compute` requires anchors. If the planner forgot, attach
        # the anchors carried on the request to keep the plan executable.
        inputs = dict(n.get("inputs") or {})
        if kind is PlanNodeKind.PRISM:
            if "pair" not in inputs:
                if not q.anchors or len(q.anchors) < 2:
                    raise GuardrailTripped(
                        "prism.compute requires at least two anchors but none were provided"
                    )
                start = 2 * prism_idx
                if len(q.anchors) >= start + 2:
                    pair = AnchorPair(a=q.anchors[start], b=q.anchors[start + 1])
                else:
                    pair = AnchorPair(a=q.anchors[0], b=q.anchors[1])
                inputs["pair"] = pair.model_dump(mode="json")
            prism_idx += 1
        nodes.append(PlanNode(
            id=n["id"],
            kind=kind,
            deps=tuple(n.get("deps") or ()),
            inputs=inputs,
            expected_tokens=int(n.get("expected_tokens", 1500)),
            confidence_prior=float(n.get("confidence_prior", 0.5)),
            rationale=str(n.get("rationale", "")),
        ))
    plan = PlanGraph(nodes=tuple(nodes), rationale=str(payload.get("rationale", "")))
    # cycle / dep validation runs on first call to topo_layers
    _ = plan.topo_layers()
    if sum(n.expected_tokens for n in nodes) > q.budget.max_tokens:
        raise PlanInfeasible("planned tokens exceed budget")
    return plan
