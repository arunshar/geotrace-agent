"""Hard physical bounds. The validator MUST reject infeasible regions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.agents.validator import ValidatorAgent
from app.config import Settings
from app.errors import KinematicViolation
from app.models import RendezvousRegion


@pytest.mark.asyncio
async def test_validator_rejects_reversed_time_window() -> None:
    settings = Settings()  # type: ignore[call-arg]
    v = ValidatorAgent(settings)
    t0 = datetime(2026, 1, 15, 6, tzinfo=UTC)
    region = RendezvousRegion(
        polygon_geojson={"type": "Point", "coordinates": [-162.0, 56.0]},
        earliest_meet_t=t0 + timedelta(hours=1),
        latest_meet_t=t0,
        confidence=0.5,
        method="TGARD",
    )
    with pytest.raises(KinematicViolation):
        await v.validate([region], domain="vessel")
