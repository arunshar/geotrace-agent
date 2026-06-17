"""Tests for the Temporal durable-orchestration layer.

Two layers of test:
  - Pure helpers (no server): the deterministic confidence/region math the
    workflow runs inline.
  - Workflow behavior (Temporal time-skipping test env, mocked activities): the
    full GeoTraceWorkflow drives a 2-node plan through the worker, proving the
    high-confidence path returns the synthesized answer and the low-confidence
    path durably waits for the human-review signal and then ships the corrected
    answer. Activities are mocked, so no LLM, DB, or live cluster is needed; the
    workflow logic and the determinism boundary are what is under test.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.models import (
    PlanGraph,
    PlanNode,
    PlanNodeKind,
    QueryIn,
    RendezvousRegion,
)
from app.temporal.models import NodeResult, ReviewDecision
from app.temporal.workflows import GeoTraceWorkflow


def _region(conf: float = 0.9) -> RendezvousRegion:
    now = datetime(2024, 1, 1, tzinfo=UTC)
    return RendezvousRegion(
        polygon_geojson={"type": "Polygon", "coordinates": []},
        earliest_meet_t=now,
        latest_meet_t=now,
        confidence=conf,
        method="TGARD",
    )


# --------------------------------------------------------------- pure helpers


def test_aggregate_confidence_no_regions_uses_priors() -> None:
    plan = PlanGraph(
        nodes=(PlanNode(id="v1", kind=PlanNodeKind.VALIDATE, confidence_prior=0.4),),
        rationale="x",
    )
    assert GeoTraceWorkflow._aggregate_confidence(plan, {}) == pytest.approx(0.4)


def test_aggregate_confidence_blends_region_and_prior() -> None:
    plan = PlanGraph(
        nodes=(PlanNode(id="v1", kind=PlanNodeKind.VALIDATE, confidence_prior=0.4),),
        rationale="x",
    )
    results = {"v1": NodeResult(node_id="v1", kind="validate.kinematic", regions=[_region(0.9)])}
    # 0.5 * 0.9 + 0.5 * 0.4 = 0.65
    assert GeoTraceWorkflow._aggregate_confidence(plan, results) == pytest.approx(0.65)


def test_collect_regions_flattens() -> None:
    results = {
        "a": NodeResult(node_id="a", kind="prism.compute"),
        "b": NodeResult(node_id="b", kind="validate.kinematic", regions=[_region(), _region()]),
    }
    assert len(GeoTraceWorkflow._collect_regions(results)) == 2


# ----------------------------------------------------------- fake orchestrator


class _FakeOrch:
    """Duck-typed stand-in exposing only the component attributes the activities use."""

    def __init__(self, *, validate_regions: list[RendezvousRegion]) -> None:
        self._validate_regions = validate_regions
        self.guard = SimpleNamespace(check=self._noop_async)
        self.out_filter = SimpleNamespace(scrub=lambda answer: None)
        self.planner = SimpleNamespace(
            plan=self._plan, last_tokens_in=10, last_tokens_out=20,
            last_cost_usd=0.001, last_cache_hit=False,
        )
        self.st_reasoner = SimpleNamespace(compute=self._compute)
        self.gap_detector = SimpleNamespace(
            detect=self._noop_async, last_tokens_in=0, last_tokens_out=0, last_cost_usd=0.0)
        self.rendezvous = SimpleNamespace(find=self._noop_async)
        self.validator = SimpleNamespace(validate=self._validate)
        self.cache = SimpleNamespace(retrieve=self._noop_async)
        self.token_opt = SimpleNamespace(summarize=self._summarize)
        self.hitl = SimpleNamespace(enqueue=self._noop_async)

    async def _noop_async(self, *a, **k):
        return None

    async def _plan(self, q, convo):
        # PRISM(p1) -> VALIDATE(v1). Both priors 0.4.
        p1 = PlanNode(
            id="p1", kind=PlanNodeKind.PRISM, confidence_prior=0.4,
            inputs={"pair": {
                "a": {"lat": 1.0, "lon": 2.0, "t": "2024-01-01T00:00:00Z"},
                "b": {"lat": 1.5, "lon": 2.5, "t": "2024-01-01T02:00:00Z"},
            }},
        )
        v1 = PlanNode(id="v1", kind=PlanNodeKind.VALIDATE, deps=("p1",), confidence_prior=0.4)
        return PlanGraph(nodes=(p1, v1), rationale="test plan")

    async def _compute(self, pair, domain):
        return SimpleNamespace(prism={"semi_major_m": 1000.0})

    async def _validate(self, cands, domain):
        return list(self._validate_regions)

    async def _summarize(self, *, question, plan, results, budget_tokens):
        return ("draft answer", 5, 30, 0.002)


def _build_worker(env, fake):
    from temporalio.worker import Worker

    from app.temporal.activities import GeoTraceActivities

    acts = GeoTraceActivities(fake)
    return Worker(
        env.client,
        task_queue="geotrace-test",
        workflows=[GeoTraceWorkflow],
        activities=[
            acts.guard, acts.plan, acts.execute_node,
            acts.summarize, acts.output_filter, acts.hitl_enqueue,
        ],
    )


@pytest.mark.asyncio
async def test_high_confidence_no_hitl() -> None:
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.testing import WorkflowEnvironment

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        fake = _FakeOrch(validate_regions=[_region(0.9)])
        async with _build_worker(env, fake):
            handle = await env.client.start_workflow(
                GeoTraceWorkflow.run,
                QueryIn(question="Where could vessel A and vessel B have met?"),
                id="wf-hi", task_queue="geotrace-test",
            )
            result = await handle.result()
    assert result.hitl_required is False
    assert result.answer == "draft answer"
    assert result.confidence == pytest.approx(0.65)


@pytest.mark.asyncio
async def test_low_confidence_waits_for_human_review_signal() -> None:
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.testing import WorkflowEnvironment

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        fake = _FakeOrch(validate_regions=[])  # no regions -> confidence 0.4 -> HITL
        async with _build_worker(env, fake):
            handle = await env.client.start_workflow(
                GeoTraceWorkflow.run,
                QueryIn(question="Where could vessel A and vessel B have met?"),
                id="wf-lo", task_queue="geotrace-test",
            )
            # The run parks on the durable review wait; a reviewer signals it.
            await handle.signal(
                GeoTraceWorkflow.review,
                ReviewDecision(approved=True, corrected_answer="reviewed answer"),
            )
            result = await handle.result()
    assert result.hitl_required is True
    assert result.answer == "reviewed answer"
