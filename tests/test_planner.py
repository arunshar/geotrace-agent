"""Planner emits a topo-sortable DAG within budget."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.agents.planner import PlannerAgent
from app.config import get_settings
from app.models import Anchor, Budget, ConversationState, PlanNodeKind, QueryIn
from app.services.semantic_cache import SemanticCache
from app.services.token_optimizer import TokenOptimizer


@pytest.mark.asyncio
async def test_planner_emits_valid_plan() -> None:
    s = get_settings()
    cache = await SemanticCache.connect(s)
    try:
        opt = TokenOptimizer(s, cache=cache)
        p = PlannerAgent(s, opt)
        q = QueryIn(
            question="Compute the prism between two anchors and find rendezvous regions",
            domain="vessel",
            anchors=[
                Anchor(lat=56.10, lon=-162.05, t=datetime(2026, 1, 15, 6, tzinfo=UTC)),
                Anchor(lat=56.30, lon=-162.40, t=datetime(2026, 1, 15, 12, tzinfo=UTC)),
            ],
            budget=Budget(),
        )
        plan = await p.plan(q, ConversationState())
        layers = plan.topo_layers()
        assert layers
        assert sum(n.expected_tokens for n in plan.nodes) <= q.budget.max_tokens
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_planner_assigns_sequential_anchor_pairs_for_rendezvous() -> None:
    s = get_settings()
    cache = await SemanticCache.connect(s)
    try:
        opt = TokenOptimizer(s, cache=cache)
        p = PlannerAgent(s, opt)
        anchors = [
            Anchor(lat=56.10, lon=-162.05, t=datetime(2026, 1, 15, 6, tzinfo=UTC)),
            Anchor(lat=56.30, lon=-162.40, t=datetime(2026, 1, 15, 12, tzinfo=UTC)),
            Anchor(lat=56.12, lon=-162.08, t=datetime(2026, 1, 15, 6, tzinfo=UTC)),
            Anchor(lat=56.28, lon=-162.34, t=datetime(2026, 1, 15, 12, tzinfo=UTC)),
        ]
        q = QueryIn(
            question="Could the two vessels rendezvous near the Aleutian shelf?",
            domain="vessel",
            anchors=anchors,
            budget=Budget(),
        )
        plan = await p.plan(q, ConversationState())
        prism_nodes = [n for n in plan.nodes if n.kind is PlanNodeKind.PRISM]

        assert len(prism_nodes) == 2
        assert prism_nodes[0].inputs["pair"]["a"]["lat"] == anchors[0].lat
        assert prism_nodes[0].inputs["pair"]["b"]["lat"] == anchors[1].lat
        assert prism_nodes[1].inputs["pair"]["a"]["lat"] == anchors[2].lat
        assert prism_nodes[1].inputs["pair"]["b"]["lat"] == anchors[3].lat
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_planner_uses_gap_detector_for_gap_question_without_anchors() -> None:
    s = get_settings()
    cache = await SemanticCache.connect(s)
    try:
        opt = TokenOptimizer(s, cache=cache)
        p = PlannerAgent(s, opt)
        q = QueryIn(
            question="Did VESSEL-1234 have a coverage gap consistent with signal denial?",
            domain="vessel",
            anchors=None,
            budget=Budget(),
        )
        plan = await p.plan(q, ConversationState())

        assert [n.kind for n in plan.nodes] == [PlanNodeKind.GAPS, PlanNodeKind.SUMMARIZE]
        assert "trajectory" in plan.nodes[0].inputs
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_planner_keeps_prism_only_question_minimal() -> None:
    s = get_settings()
    cache = await SemanticCache.connect(s)
    try:
        opt = TokenOptimizer(s, cache=cache)
        p = PlannerAgent(s, opt)
        q = QueryIn(
            question="Compute the prism between the two anchors.",
            domain="vessel",
            anchors=[
                Anchor(lat=56.10, lon=-162.05, t=datetime(2026, 1, 15, 6, tzinfo=UTC)),
                Anchor(lat=56.30, lon=-162.40, t=datetime(2026, 1, 15, 12, tzinfo=UTC)),
            ],
            budget=Budget(),
        )
        plan = await p.plan(q, ConversationState())

        assert [n.kind for n in plan.nodes] == [PlanNodeKind.PRISM, PlanNodeKind.SUMMARIZE]
    finally:
        await cache.close()
