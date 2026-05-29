"""Typed request / response / state objects. Pydantic v2."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Mutable(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Public schemas ---------------------------------------------------------


class HealthOut(_Frozen):
    status: Literal["ok", "degraded"] = "ok"
    version: str


class Budget(_Frozen):
    max_tokens: int = Field(12_000, ge=500, le=200_000)
    max_tools: int = Field(8, ge=1, le=64)
    max_seconds: float = Field(30.0, gt=0, le=300)


class Anchor(_Frozen):
    """A space-time anchor (x, y, t)."""

    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    t: datetime

    @field_validator("t")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return v.astimezone(UTC) if v.tzinfo else v.replace(tzinfo=UTC)


class AnchorPair(_Frozen):
    a: Anchor
    b: Anchor


class SpeedBounds(_Frozen):
    v_max_mps: float = Field(..., gt=0)
    v_min_mps: float = 0.0
    domain: Literal["vessel", "vehicle", "pedestrian", "uav"] = "vessel"


class QueryIn(_Mutable):
    question: str = Field(..., min_length=4, max_length=2_000)
    budget: Budget = Budget()
    conversation_id: UUID | None = None
    anchors: list[Anchor] | None = None
    domain: Literal["vessel", "vehicle", "pedestrian", "uav"] = "vessel"


class Citation(_Frozen):
    source: str
    span: tuple[int, int] | None = None
    score: float | None = None


class GeoEllipse(_Frozen):
    """Ellipse on the (lat, lon) plane: locus of d(p, A) + d(p, B) <= L."""

    a_lat: float
    a_lon: float
    b_lat: float
    b_lon: float
    semi_major_m: float
    semi_minor_m: float
    rotation_rad: float


class RendezvousRegion(_Frozen):
    polygon_geojson: dict[str, Any]
    earliest_meet_t: datetime
    latest_meet_t: datetime
    confidence: float = Field(..., ge=0, le=1)
    method: Literal["TGARD", "DC-TGARD", "STP", "STAGD", "STP-baseline"]


class StageTrace(_Frozen):
    name: str
    started_at: datetime
    ended_at: datetime
    tokens_in: int
    tokens_out: int
    cost_usd: float
    cache_hit: bool = False


class QueryOut(_Mutable):
    answer: str
    regions: list[RendezvousRegion] = []
    citations: list[Citation] = []
    confidence: float = Field(..., ge=0, le=1)
    terminated_by_budget: bool = False
    trace_id: str
    stages: list[StageTrace] = []
    tokens_total: int
    cost_usd_total: float
    hitl_required: bool = False


class FeedbackIn(_Mutable):
    trace_id: str
    label: Literal["correct", "incorrect", "ambiguous"]
    notes: str | None = None
    reviewer: str


class FeedbackOut(_Frozen):
    accepted: bool = True
    queue_position: int | None = None


# --- Internal state ---------------------------------------------------------


class PlanNodeKind(StrEnum):
    PRISM = "prism.compute"
    GAPS = "gaps.detect"
    TGARD = "rendezvous.tgard"
    DC_TGARD = "rendezvous.dc_tgard"
    VALIDATE = "validate.kinematic"
    RETRIEVE = "retrieve.semantic"
    SUMMARIZE = "summarize"


class PlanNode(_Frozen):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    kind: PlanNodeKind
    inputs: dict[str, Any] = {}
    deps: tuple[str, ...] = ()
    expected_tokens: int = 1500
    confidence_prior: float = 0.5
    rationale: str = ""


class PlanGraph(_Frozen):
    nodes: tuple[PlanNode, ...]
    rationale: str

    def topo_layers(self) -> list[list[PlanNode]]:
        """Return layers of nodes safe to run in parallel."""

        remaining = {n.id: n for n in self.nodes}
        done: set[str] = set()
        layers: list[list[PlanNode]] = []
        while remaining:
            ready = [n for n in remaining.values() if all(d in done for d in n.deps)]
            if not ready:
                raise ValueError("plan graph has a cycle")
            layers.append(ready)
            for n in ready:
                done.add(n.id)
                del remaining[n.id]
        return layers


class ConversationState(_Mutable):
    conversation_id: UUID = Field(default_factory=uuid4)
    history: list[dict[str, Any]] = []
    summary: str = ""
