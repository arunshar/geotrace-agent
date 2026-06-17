"""GeoTraceWorkflow: the durable, deterministic control plane for a GeoTrace run.

This is a faithful port of app/services/orchestrator.Orchestrator.run. The line
between this file and activities.py is the project's neuro-symbolic boundary made
literal, and it is exactly Temporal's determinism boundary:

  workflow (here)   deterministic symbolic control: the plan-graph topo schedule,
                    the token / tool / wallclock budget guard, confidence
                    aggregation, and the human-in-the-loop gate. Replays from the
                    event history and must make the same decisions every time, so
                    it uses workflow.now() and workflow.uuid4() instead of the
                    wall clock and random, and does no I/O.
  activities        every nondeterministic or side-effecting step: the LLM
                    planner / gap detector / summarizer, the symbolic geo kernels,
                    the cache read, and the HITL enqueue. Recorded once, replayed
                    verbatim.

Low confidence triggers the durable-agent pattern: the workflow enqueues for
review and then DURABLY WAITS on a human-approval signal (wait_condition), which
survives worker restarts for as long as a human takes, instead of the
orchestrator's fire-and-forget enqueue.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from app.models import (
        PlanGraph,
        PlanNodeKind,
        QueryIn,
        QueryOut,
        RendezvousRegion,
    )
    from app.temporal.activities import GeoTraceActivities
    from app.temporal.models import NodeResult, ReviewDecision

# Below this confidence a run is routed to human review. In production the worker
# injects this from Settings.hitl_confidence_threshold via a workflow argument;
# kept a constant here so the workflow stays free of the settings import.
HITL_CONFIDENCE_THRESHOLD = 0.6

# Activity options. at-least-once delivery plus a bounded retry; every activity is
# idempotent (see activities.py). LLM activities get a longer deadline.
_LLM = dict(
    start_to_close_timeout=timedelta(seconds=120),
    retry_policy=RetryPolicy(maximum_attempts=3),
)
_FAST = dict(
    start_to_close_timeout=timedelta(seconds=30),
    retry_policy=RetryPolicy(maximum_attempts=3),
)


@workflow.defn(name="GeoTraceWorkflow")
class GeoTraceWorkflow:
    def __init__(self) -> None:
        self._review: ReviewDecision | None = None
        self._stage: str = "init"
        self._tokens: int = 0
        self._tools: int = 0
        self._cost: float = 0.0

    # ----------------------------------------------------------------- run

    @workflow.run
    async def run(self, q: QueryIn) -> QueryOut:
        trace_id = workflow.uuid4().hex
        start = workflow.now()

        # input guard
        self._stage = "guard"
        await workflow.execute_activity_method(
            GeoTraceActivities.guard, q.question, **_FAST)

        # planning (LLM, nondeterministic): recorded once, replayed verbatim
        self._stage = "plan"
        pr = await workflow.execute_activity_method(
            GeoTraceActivities.plan, q, **_LLM)
        plan: PlanGraph = pr.plan
        self._account(pr.tokens_in, pr.tokens_out, pr.cost_usd)
        self._enforce_budget(q, start)

        # execute the plan graph layer by layer; each topo layer fans out in
        # parallel, the same shape as the orchestrator's asyncio.gather(per layer)
        results: dict[str, NodeResult] = {}
        for layer in plan.topo_layers():
            self._stage = "execute:" + ",".join(n.kind.value for n in layer)
            futs = [
                workflow.execute_activity_method(
                    GeoTraceActivities.execute_node,
                    args=[node, q, [results[d] for d in node.deps if d in results]],
                    **(_LLM if node.kind is PlanNodeKind.GAPS else _FAST),
                )
                for node in layer
            ]
            outs: list[NodeResult] = list(await asyncio.gather(*futs))
            for node, out in zip(layer, outs, strict=True):
                self._tools += 1
                self._account(out.tokens_in, out.tokens_out, out.cost_usd)
                if out.violation:
                    # hard stop: a kinematically impossible answer must not ship
                    raise ApplicationError(
                        f"kinematic violation at node {node.id}", non_retryable=True)
                results[node.id] = out
            self._enforce_budget(q, start)

        # synthesis (LLM, nondeterministic)
        self._stage = "summarize"
        tokens_left = q.budget.max_tokens - self._tokens
        sr = await workflow.execute_activity_method(
            GeoTraceActivities.summarize,
            args=[plan, list(results.values()), q, tokens_left], **_LLM)
        self._account(sr.tokens_in, sr.tokens_out, sr.cost_usd)

        regions = self._collect_regions(results)
        confidence = self._aggregate_confidence(plan, results)

        # output guard (scrub PII / unsafe content)
        self._stage = "output_filter"
        answer = await workflow.execute_activity_method(
            GeoTraceActivities.output_filter, sr.answer, **_FAST)

        # human-in-the-loop gate: enqueue, then DURABLY WAIT for the review signal
        hitl_required = confidence < HITL_CONFIDENCE_THRESHOLD
        if hitl_required:
            self._stage = "hitl_enqueue"
            await workflow.execute_activity_method(
                GeoTraceActivities.hitl_enqueue,
                args=[trace_id, q, answer, regions], **_FAST)
            self._stage = "awaiting_human_review"
            await workflow.wait_condition(lambda: self._review is not None)
            if self._review is not None and self._review.corrected_answer:
                answer = self._review.corrected_answer

        self._stage = "done"
        return QueryOut(
            answer=answer,
            regions=regions,
            citations=[],
            confidence=confidence,
            terminated_by_budget=False,
            trace_id=trace_id,
            stages=[],
            tokens_total=self._tokens,
            cost_usd_total=self._cost,
            hitl_required=hitl_required,
        )

    # ----------------------------------------------------------- signal/query

    @workflow.signal
    async def review(self, decision: ReviewDecision) -> None:
        """Human reviewer's verdict; releases the durable wait above."""
        self._review = decision

    @workflow.query
    def progress(self) -> dict:
        """Synchronous read of live run state (no mutation)."""
        return {
            "stage": self._stage,
            "tokens": self._tokens,
            "tools": self._tools,
            "cost_usd": self._cost,
            "reviewed": self._review is not None,
        }

    # -------------------------------------------------------------- helpers

    def _account(self, tokens_in: int, tokens_out: int, cost: float) -> None:
        self._tokens += tokens_in + tokens_out
        self._cost += cost

    def _enforce_budget(self, q: QueryIn, start) -> None:
        """Deterministic budget guard. Uses workflow.now(), never the wall clock."""
        if self._tokens > q.budget.max_tokens:
            raise ApplicationError(
                f"budget: tokens {self._tokens} > {q.budget.max_tokens}",
                non_retryable=True)
        if self._tools > q.budget.max_tools:
            raise ApplicationError(
                f"budget: tools {self._tools} > {q.budget.max_tools}",
                non_retryable=True)
        elapsed = (workflow.now() - start).total_seconds()
        if elapsed > q.budget.max_seconds:
            raise ApplicationError("budget: wallclock exceeded", non_retryable=True)

    @staticmethod
    def _collect_regions(results: dict[str, NodeResult]) -> list[RendezvousRegion]:
        # Rebuild each region through THIS sandbox's RendezvousRegion class.
        # Regions inside a NodeResult arrive deserialized by the data converter in
        # the activity worker, whose RendezvousRegion class object differs from the
        # workflow sandbox's, so QueryOut's list[RendezvousRegion] validation would
        # reject the foreign-but-identical instances ("Input should be ... an
        # instance of RendezvousRegion"). Dump-then-validate re-anchors the class
        # identity to the one QueryOut expects.
        out: list[RendezvousRegion] = []
        for r in results.values():
            for region in r.regions:
                data = region.model_dump() if hasattr(region, "model_dump") else region
                out.append(RendezvousRegion.model_validate(data))
        return out

    @staticmethod
    def _aggregate_confidence(plan: PlanGraph, results: dict[str, NodeResult]) -> float:
        """Same aggregation as Orchestrator._aggregate_confidence, on NodeResults."""
        priors = [n.confidence_prior for n in plan.nodes]
        regions: list[RendezvousRegion] = []
        for r in results.values():
            regions.extend(r.regions)
        if regions:
            return min(
                0.99,
                0.5 * sum(r.confidence for r in regions) / len(regions)
                + 0.5 * (sum(priors) / max(1, len(priors))),
            )
        return min(0.99, sum(priors) / max(1, len(priors)))
