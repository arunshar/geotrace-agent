"""HITL queue. Mirrors Centific's HITL pattern from ContraGen / LegalWiz / ART.

Backed by a Postgres table; read by the Streamlit reviewer console in
`frontend/app.py`. The reviewer's verdict can later be exported as a
preference dataset for the sibling `pi-grpo` project's DPO trainer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog

from app.config import Settings
from app.models import FeedbackIn, PlanGraph, QueryIn, RendezvousRegion

log = structlog.get_logger(__name__)


class HitlQueue:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._mem: list[dict[str, Any]] = []

    @classmethod
    async def connect(cls, settings: Settings) -> HitlQueue:
        return cls(settings)

    async def close(self) -> None:
        return None

    async def enqueue(
        self,
        trace_id: str,
        q: QueryIn,
        plan: PlanGraph,
        results: dict[str, Any],
        regions: list[RendezvousRegion],
    ) -> None:
        item = {
            "trace_id": trace_id,
            "question": q.question,
            "plan_rationale": plan.rationale,
            "n_regions": len(regions),
            "regions": [r.model_dump(mode="json") for r in regions],
            "ts": datetime.now(UTC).isoformat(),
        }
        self._mem.append(item)
        log.info("hitl_enqueue", **{k: v for k, v in item.items() if k != "regions"})

    async def label(self, payload: FeedbackIn) -> int:
        log.info("hitl_label", **payload.model_dump())
        # In production: update the row in postgres and return its
        # remaining position in the queue.
        return max(0, len(self._mem) - 1)
