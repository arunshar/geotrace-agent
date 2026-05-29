"""Orchestrator smoke test. Exercises the planner + agents end to end with stubbed LLMs."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.config import get_settings
from app.models import Anchor, Budget, QueryIn
from app.services.orchestrator import Orchestrator


@pytest.mark.asyncio
async def test_smoke_runs_with_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GT_ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("GT_OPENAI_API_KEY", "")
    get_settings.cache_clear()
    s = get_settings()
    orch = await Orchestrator.bootstrap(s)
    try:
        out = await orch.run(QueryIn(
            question="Could a vessel near 56N 162W rendezvous between 06:00Z and 12:00Z?",
            domain="vessel",
            anchors=[
                Anchor(lat=56.10, lon=-162.05, t=datetime(2026, 1, 15, 6, tzinfo=UTC)),
                Anchor(lat=56.30, lon=-162.40, t=datetime(2026, 1, 15, 12, tzinfo=UTC)),
            ],
            budget=Budget(),
        ))
        assert out.trace_id
        assert out.tokens_total >= 0
        assert out.confidence >= 0.0
    finally:
        await orch.shutdown()
