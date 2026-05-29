"""Per-trace cost ledger.

Writes a row per (trace_id, stage_name) into Postgres. Streamed to a
Grafana board so SREs can answer "what does a query cost?" without
parsing OTEL traces.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from app.config import Settings

log = structlog.get_logger(__name__)


class CostTracker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # In production: asyncpg.Pool. The scaffold logs structured rows.

    def record(
        self,
        *,
        trace_id: str,
        stage: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        cache_hit: bool,
    ) -> None:
        log.info(
            "cost",
            trace_id=trace_id,
            stage=stage,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            cache_hit=cache_hit,
            ts=datetime.now(UTC).isoformat(),
        )
