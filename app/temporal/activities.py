"""Activities: every nondeterministic or side-effecting step of a GeoTrace run.

An activity runs AT-LEAST-ONCE (Temporal redelivers after a worker crash or a
visibility timeout), so each one must be idempotent. The LLM activities (plan,
gap detection, summarize) are nondeterministic, which is exactly why they are
activities and not workflow code: their result is recorded in history once and
replayed verbatim, so a workflow replay never re-calls the model and never gets a
different answer. The symbolic activities (prism, rendezvous, validate) are
deterministic; they live here only because they import heavy numeric/geo code
that must stay out of the deterministic workflow sandbox, not because they are
nondeterministic.

These wrap the existing app/services/orchestrator components, so the durable port
reuses the real agents rather than reimplementing them. A GeoTraceActivities
instance holds a bootstrapped Orchestrator and is registered with the worker; the
deterministic test registers a stand-in instead.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from temporalio import activity

from app.models import (
    AnchorPair,
    PlanGraph,
    PlanNode,
    PlanNodeKind,
    QueryIn,
    RendezvousRegion,
)
from app.temporal.models import NodeResult, PlanResult, SummaryResult


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of an agent output into a serializable payload.

    pydantic model -> dict, dataclass -> dict, else returned as is. Keeps the
    cross-activity handoff (for example a PRISM node's prism consumed by a TGARD
    node) flat and version-stable.
    """

    if obj is None or isinstance(obj, (str, int, float, bool, dict, list)):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return str(obj)


def _regions(obj: Any) -> list[RendezvousRegion]:
    """Pull RendezvousRegion objects out of an agent output (list or single)."""

    if isinstance(obj, RendezvousRegion):
        return [obj]
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, RendezvousRegion)]
    return []


class GeoTraceActivities:
    """Activity implementations bound to a live Orchestrator's components."""

    def __init__(self, orch: Any) -> None:
        # `orch` is an app.services.orchestrator.Orchestrator (or, in tests, a
        # stand-in exposing the same component attributes). Kept as Any so this
        # module imports without dragging in the heavy orchestrator graph.
        self._orch = orch

    # --- security -----------------------------------------------------------

    @activity.defn(name="geotrace.guard")
    async def guard(self, question: str) -> None:
        """Input guard. Idempotent: a pure check with no state mutation."""
        await self._orch.guard.check(question)

    # --- planning (LLM, nondeterministic) -----------------------------------

    @activity.defn(name="geotrace.plan")
    async def plan(self, q: QueryIn) -> PlanResult:
        planner = self._orch.planner
        # The planner takes a conversation state; a fresh run starts empty. A
        # conversation-scoped port would load prior state in its own activity.
        from app.models import ConversationState

        plan: PlanGraph = await planner.plan(q, ConversationState())
        return PlanResult(
            plan=plan,
            tokens_in=getattr(planner, "last_tokens_in", 0),
            tokens_out=getattr(planner, "last_tokens_out", 0),
            cost_usd=getattr(planner, "last_cost_usd", 0.0),
            cache_hit=getattr(planner, "last_cache_hit", False),
        )

    # --- one plan-graph node ------------------------------------------------

    @activity.defn(name="geotrace.execute_node")
    async def execute_node(
        self, node: PlanNode, q: QueryIn, prior: list[NodeResult]
    ) -> NodeResult:
        """Run one node, dispatching by kind exactly like Orchestrator._execute_node.

        Idempotent per (workflow run, node id): the symbolic kinds are pure
        functions of their inputs; the LLM kind (GAPS) is safe to retry because
        the workflow records the first successful result and never re-runs it on
        replay. `prior` are the already-serialized upstream NodeResults this node
        depends on.
        """
        res = NodeResult(node_id=node.id, kind=node.kind.value)
        try:
            if node.kind is PlanNodeKind.PRISM:
                pair = AnchorPair(**node.inputs["pair"])
                out = await self._orch.st_reasoner.compute(pair, q.domain)
                # Serialize the prism so a downstream TGARD node can consume it
                # across the activity boundary.
                res.payload = {"prism": _jsonable(getattr(out, "prism", out))}
                res.regions = _regions(out)
            elif node.kind is PlanNodeKind.GAPS:
                gd = self._orch.gap_detector
                out = await gd.detect(node.inputs)
                res.regions = _regions(out)
                res.payload = {"gaps": _jsonable(out)}
                res.tokens_in = getattr(gd, "last_tokens_in", 0)
                res.tokens_out = getattr(gd, "last_tokens_out", 0)
                res.cost_usd = getattr(gd, "last_cost_usd", 0.0)
            elif node.kind in (PlanNodeKind.TGARD, PlanNodeKind.DC_TGARD):
                prisms = [p.payload["prism"] for p in prior if "prism" in p.payload]
                method = "TGARD" if node.kind is PlanNodeKind.TGARD else "DC-TGARD"
                out = await self._orch.rendezvous.find(prisms, method=method)
                res.regions = _regions(out)
            elif node.kind is PlanNodeKind.VALIDATE:
                cands: list[RendezvousRegion] = []
                for p in prior:
                    cands.extend(p.regions)
                out = await self._orch.validator.validate(cands, domain=q.domain)
                res.regions = _regions(out)
            elif node.kind is PlanNodeKind.RETRIEVE:
                out = await self._orch.cache.retrieve(node.inputs.get("query", q.question))
                res.payload = {"retrieved": _jsonable(out)}
                res.cache_hit = True
            elif node.kind is PlanNodeKind.SUMMARIZE:
                pass  # the final synthesis is its own activity (see summarize)
            else:  # pragma: no cover
                raise ValueError(f"unknown node kind: {node.kind}")
        except Exception as exc:
            # A kinematic violation is a hard stop; everything else is a soft
            # per-node error the workflow records and continues past.
            from app.errors import KinematicViolation

            if isinstance(exc, KinematicViolation):
                res.violation = True
            res.error = str(exc)
        return res

    # --- synthesis (LLM, nondeterministic) ----------------------------------

    @activity.defn(name="geotrace.summarize")
    async def summarize(
        self, plan: PlanGraph, results: list[NodeResult], q: QueryIn, tokens_left: int
    ) -> SummaryResult:
        # Rebuild the {node_id: payload} shape the token optimizer expects.
        result_map: dict[str, Any] = {r.node_id: (r.payload or r.regions) for r in results}
        answer, tin, tout, cost = await self._orch.token_opt.summarize(
            question=q.question, plan=plan, results=result_map,
            budget_tokens=max(0, tokens_left),
        )
        return SummaryResult(answer=answer, tokens_in=tin, tokens_out=tout, cost_usd=cost)

    # --- output guard -------------------------------------------------------

    @activity.defn(name="geotrace.output_filter")
    async def output_filter(self, answer: str) -> str:
        """Scrub the answer. Idempotent: scrubbing twice equals scrubbing once."""
        self._orch.out_filter.scrub(answer)
        return answer

    # --- side effect: human-in-the-loop enqueue -----------------------------

    @activity.defn(name="geotrace.hitl_enqueue")
    async def hitl_enqueue(
        self, trace_id: str, q: QueryIn, answer: str, regions: list[RendezvousRegion]
    ) -> None:
        """Enqueue a low-confidence run for human review.

        At-least-once delivery means this can fire twice on a retry, so the queue
        write must be idempotent on trace_id (the durable workflow id is the
        natural dedup key). The workflow then waits on the review signal.
        """
        await self._orch.hitl.enqueue(trace_id, q, None, regions, regions)
