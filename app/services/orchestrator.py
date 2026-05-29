"""Multi-agent orchestrator. Topo-sorts the plan graph and runs nodes in parallel.

Enforces the global token / tool / wallclock budget. Every stage emits an
OpenTelemetry span and a structured cost record. Mirrors the agent
specialization pattern from Centific's ContraGen / LegalWiz frameworks.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog

from app.agents.gap_detector import GapDetectorAgent
from app.agents.planner import PlannerAgent
from app.agents.rendezvous_finder import RendezvousFinderAgent
from app.agents.space_time_reasoner import SpaceTimeReasoner
from app.agents.validator import ValidatorAgent
from app.config import Settings, get_settings
from app.errors import BudgetExceeded, KinematicViolation
from app.models import (
    AnchorPair,
    Budget,
    ConversationState,
    FeedbackIn,
    FeedbackOut,
    PlanGraph,
    PlanNode,
    PlanNodeKind,
    QueryIn,
    QueryOut,
    RendezvousRegion,
    StageTrace,
)
from app.security.input_guard import InputGuard
from app.security.output_filter import OutputFilter
from app.services.semantic_cache import SemanticCache
from app.services.token_optimizer import TokenOptimizer
from observability.cost_tracker import CostTracker
from observability.feedback import HitlQueue
from observability.tracer import span

log = structlog.get_logger(__name__)


@dataclass
class _RunCtx:
    trace_id: str
    started_at: float
    tokens: int = 0
    tools: int = 0
    cost_usd: float = 0.0
    stages: list[StageTrace] = None  # type: ignore[assignment]
    terminated_by_budget: bool = False

    def __post_init__(self) -> None:
        if self.stages is None:
            self.stages = []


class Orchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        cache: SemanticCache,
        token_opt: TokenOptimizer,
        planner: PlannerAgent,
        st_reasoner: SpaceTimeReasoner,
        gap_detector: GapDetectorAgent,
        rendezvous: RendezvousFinderAgent,
        validator: ValidatorAgent,
        guard: InputGuard,
        out_filter: OutputFilter,
        cost: CostTracker,
        hitl: HitlQueue,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self.token_opt = token_opt
        self.planner = planner
        self.st_reasoner = st_reasoner
        self.gap_detector = gap_detector
        self.rendezvous = rendezvous
        self.validator = validator
        self.guard = guard
        self.out_filter = out_filter
        self.cost = cost
        self.hitl = hitl

    # ------------------------------------------------------------- bootstrap

    @classmethod
    async def bootstrap(cls, settings: Settings | None = None) -> Orchestrator:
        s = settings or get_settings()
        cache = await SemanticCache.connect(s)
        token_opt = TokenOptimizer(s, cache=cache)
        planner = PlannerAgent(s, token_opt)
        st = SpaceTimeReasoner(s)
        gd = GapDetectorAgent(s, token_opt)
        rdv = RendezvousFinderAgent(s, st)
        val = ValidatorAgent(s)
        guard = InputGuard(s)
        outf = OutputFilter(s)
        cost = CostTracker(s)
        hitl = await HitlQueue.connect(s)
        return cls(
            settings=s, cache=cache, token_opt=token_opt, planner=planner,
            st_reasoner=st, gap_detector=gd, rendezvous=rdv, validator=val,
            guard=guard, out_filter=outf, cost=cost, hitl=hitl,
        )

    async def shutdown(self) -> None:
        await asyncio.gather(self.cache.close(), self.hitl.close(), return_exceptions=True)

    def capability_card(self) -> dict[str, Any]:
        return {
            "name": "geotrace-agent",
            "version": self.settings.version,
            "capabilities": [
                "plan.decompose",
                "prism.compute",
                "rendezvous.candidates",
                "gaps.detect",
                "rendezvous.tgard",
                "rendezvous.dc_tgard",
                "validate.kinematic",
            ],
            "a2a_endpoint": "/a2a/jsonrpc",
            "models": {"primary": self.settings.primary_model, "fallback": self.settings.fallback_model},
        }

    # ----------------------------------------------------------------- run

    async def run(self, q: QueryIn) -> QueryOut:
        ctx = _RunCtx(trace_id=uuid4().hex, started_at=time.monotonic())
        await self.guard.check(q.question)

        with span("orchestrator.run", attributes={"trace_id": ctx.trace_id, "domain": q.domain}):
            convo = ConversationState(conversation_id=q.conversation_id) if q.conversation_id else ConversationState()
            plan = await self._planning_stage(q, convo, ctx)
            results = await self._execute_plan(plan, q, ctx)
            answer = await self._summarize(plan, results, q, ctx)
            regions = self._collect_regions(results)
            confidence = self._aggregate_confidence(plan, results)
            self.out_filter.scrub(answer)

            hitl_required = confidence < self.settings.hitl_confidence_threshold
            if hitl_required:
                await self.hitl.enqueue(ctx.trace_id, q, plan, results, regions)

            return QueryOut(
                answer=answer,
                regions=regions,
                citations=[],
                confidence=confidence,
                terminated_by_budget=ctx.terminated_by_budget,
                trace_id=ctx.trace_id,
                stages=ctx.stages,
                tokens_total=ctx.tokens,
                cost_usd_total=ctx.cost_usd,
                hitl_required=hitl_required,
            )

    # -------------------------------------------------------------- stages

    async def _planning_stage(self, q: QueryIn, convo: ConversationState, ctx: _RunCtx) -> PlanGraph:
        with self._stage(ctx, "planner.plan") as rec:
            plan = await self.planner.plan(q, convo)
            rec.tokens_in = self.planner.last_tokens_in
            rec.tokens_out = self.planner.last_tokens_out
            rec.cost_usd = self.planner.last_cost_usd
            rec.cache_hit = self.planner.last_cache_hit
        ctx.tokens += rec.tokens_in + rec.tokens_out
        ctx.cost_usd += rec.cost_usd
        self._enforce_budget(q.budget, ctx)
        return plan

    async def _execute_plan(self, plan: PlanGraph, q: QueryIn, ctx: _RunCtx) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for layer in plan.topo_layers():
            tasks = [self._execute_node(node, q, ctx, results) for node in layer]
            outs = await asyncio.gather(*tasks, return_exceptions=True)
            for node, out in zip(layer, outs, strict=True):
                if isinstance(out, KinematicViolation):
                    raise out
                if isinstance(out, Exception):
                    log.warning("node_failed", node_id=node.id, kind=node.kind, err=str(out))
                    results[node.id] = {"error": str(out)}
                else:
                    results[node.id] = out
            self._enforce_budget(q.budget, ctx)
        return results

    async def _execute_node(
        self, node: PlanNode, q: QueryIn, ctx: _RunCtx, prior: dict[str, Any]
    ) -> Any:
        ctx.tools += 1
        with self._stage(ctx, f"node.{node.kind.value}") as rec:
            if node.kind is PlanNodeKind.PRISM:
                pair = AnchorPair(**node.inputs["pair"])
                out = await self.st_reasoner.compute(pair, q.domain)
            elif node.kind is PlanNodeKind.GAPS:
                out = await self.gap_detector.detect(node.inputs)
                rec.tokens_in = self.gap_detector.last_tokens_in
                rec.tokens_out = self.gap_detector.last_tokens_out
                rec.cost_usd = self.gap_detector.last_cost_usd
            elif node.kind in (PlanNodeKind.TGARD, PlanNodeKind.DC_TGARD):
                upstream_prisms = [prior[d].prism for d in node.deps if hasattr(prior.get(d), "prism")]
                method = "TGARD" if node.kind is PlanNodeKind.TGARD else "DC-TGARD"
                out = await self.rendezvous.find(upstream_prisms, method=method)
            elif node.kind is PlanNodeKind.VALIDATE:
                cands = [r for r in prior.get(node.deps[0], []) if isinstance(r, RendezvousRegion)] if node.deps else []
                out = await self.validator.validate(cands, domain=q.domain)
            elif node.kind is PlanNodeKind.RETRIEVE:
                out = await self.cache.retrieve(node.inputs.get("query", q.question))
                rec.cache_hit = True
            elif node.kind is PlanNodeKind.SUMMARIZE:
                out = None  # handled in `_summarize`
            else:  # pragma: no cover
                raise ValueError(f"unknown node kind: {node.kind}")
        ctx.tokens += rec.tokens_in + rec.tokens_out
        ctx.cost_usd += rec.cost_usd
        return out

    async def _summarize(
        self, plan: PlanGraph, results: dict[str, Any], q: QueryIn, ctx: _RunCtx
    ) -> str:
        with self._stage(ctx, "summarize") as rec:
            answer, tokens_in, tokens_out, cost = await self.token_opt.summarize(
                question=q.question, plan=plan, results=results,
                budget_tokens=q.budget.max_tokens - ctx.tokens,
            )
            rec.tokens_in, rec.tokens_out, rec.cost_usd = tokens_in, tokens_out, cost
        ctx.tokens += tokens_in + tokens_out
        ctx.cost_usd += cost
        return answer

    def _collect_regions(self, results: dict[str, Any]) -> list[RendezvousRegion]:
        out: list[RendezvousRegion] = []
        for v in results.values():
            if isinstance(v, list):
                out.extend(r for r in v if isinstance(r, RendezvousRegion))
        return out

    @staticmethod
    def _aggregate_confidence(plan: PlanGraph, results: dict[str, Any]) -> float:
        priors = [n.confidence_prior for n in plan.nodes]
        regions: list[RendezvousRegion] = []
        for v in results.values():
            if isinstance(v, list):
                regions.extend(r for r in v if isinstance(r, RendezvousRegion))
        if regions:
            return min(0.99, 0.5 * sum(r.confidence for r in regions) / len(regions)
                       + 0.5 * (sum(priors) / max(1, len(priors))))
        return min(0.99, sum(priors) / max(1, len(priors)))

    # ------------------------------------------------------------- helpers

    def _enforce_budget(self, budget: Budget, ctx: _RunCtx) -> None:
        if ctx.tokens > budget.max_tokens:
            ctx.terminated_by_budget = True
            raise BudgetExceeded(f"tokens {ctx.tokens} > {budget.max_tokens}")
        if ctx.tools > budget.max_tools:
            ctx.terminated_by_budget = True
            raise BudgetExceeded(f"tools {ctx.tools} > {budget.max_tools}")
        if (time.monotonic() - ctx.started_at) > budget.max_seconds:
            ctx.terminated_by_budget = True
            raise BudgetExceeded("wallclock exceeded")

    class _StageRec:
        def __init__(self) -> None:
            self.tokens_in = 0
            self.tokens_out = 0
            self.cost_usd = 0.0
            self.cache_hit = False

    def _stage(self, ctx: _RunCtx, name: str) -> _StageContextManager:
        return _StageContextManager(ctx, name)

    # ----------------------------------------------------------------- HITL

    async def record_feedback(self, payload: FeedbackIn) -> FeedbackOut:
        pos = await self.hitl.label(payload)
        return FeedbackOut(accepted=True, queue_position=pos)


class _StageContextManager:
    def __init__(self, ctx: _RunCtx, name: str) -> None:
        self.ctx = ctx
        self.name = name
        self.rec = Orchestrator._StageRec()
        self._span_cm: Any | None = None
        self._t0: datetime | None = None

    def __enter__(self) -> Orchestrator._StageRec:
        self._t0 = datetime.now(UTC)
        self._span_cm = span(self.name)
        self._span_cm.__enter__()  # type: ignore[union-attr]
        return self.rec

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        t1 = datetime.now(UTC)
        if self._span_cm is not None:
            self._span_cm.__exit__(exc_type, exc, tb)
        self.ctx.stages.append(StageTrace(
            name=self.name,
            started_at=self._t0 or t1,
            ended_at=t1,
            tokens_in=self.rec.tokens_in,
            tokens_out=self.rec.tokens_out,
            cost_usd=self.rec.cost_usd,
            cache_hit=self.rec.cache_hit,
        ))
