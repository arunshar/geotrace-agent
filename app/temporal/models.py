"""Serializable data contracts that cross the Temporal activity boundary.

Every value an activity returns or a signal carries must round-trip through
Temporal's data converter (here, the pydantic converter) and be stable across
worker versions, so these are deliberately flat pydantic models rather than the
orchestrator's live in-memory objects (a Space-Time prism handle, an open netCDF
reader). The activity packs its result into one of these; the workflow only ever
sees the serialized form, which is what keeps the workflow deterministic and
replayable.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.models import PlanGraph, RendezvousRegion


class PlanResult(BaseModel):
    """Output of the planner activity (the LLM decomposition)."""

    plan: PlanGraph
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cache_hit: bool = False


class NodeResult(BaseModel):
    """Output of one plan-node activity.

    `regions` carries any rendezvous regions the node produced (the durable
    answer payload). `payload` carries the kind-specific serialized handoff a
    downstream node needs (for example a PRISM node serializes its prism here so
    a TGARD node can consume it without sharing a live object). `error` is set
    instead of raising for a soft node failure, mirroring the orchestrator's
    per-node error capture; a hard kinematic violation sets `violation=True` so
    the workflow can abort the run.
    """

    node_id: str
    kind: str
    regions: list[RendezvousRegion] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cache_hit: bool = False
    error: str | None = None
    violation: bool = False


class SummaryResult(BaseModel):
    """Output of the summarizer activity (the LLM synthesis of the final answer)."""

    answer: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class ReviewDecision(BaseModel):
    """Payload of the human-in-the-loop approval signal.

    When a run's confidence is below threshold the workflow durably waits for one
    of these (a reviewer signals approve or reject, optionally rewriting the
    answer). Surviving worker restarts while waiting is the whole point of doing
    HITL on Temporal rather than a fire-and-forget queue.
    """

    approved: bool
    corrected_answer: str | None = None
    reviewer: str = "human"
