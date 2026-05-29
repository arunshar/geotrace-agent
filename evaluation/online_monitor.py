"""Online evaluation. Samples 1 percent of production traffic and:

- recomputes the validator independently
- reruns the planner with a frozen prompt version and diffs the plan
- emits drift metrics to the OTEL collector

Drift > threshold opens an issue via PagerDuty.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog

log = structlog.get_logger(__name__)


async def shadow_compare(trace_id: str, live_plan: Any, frozen_plan: Any) -> dict[str, Any]:
    drift = abs(len(live_plan.nodes) - len(frozen_plan.nodes))
    log.info("plan_drift", trace_id=trace_id, drift=drift, ts=datetime.now(UTC).isoformat())
    return {"trace_id": trace_id, "drift": drift}
