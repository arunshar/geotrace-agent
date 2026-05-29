"""Deterministic geometric reasoner. No LLM.

Turns a `Prism` into a typed result that downstream agents (rendezvous,
gap detector, validator) can consume without re-deriving the geometry.
This is the agent that owns Arun's signature space-time prism math.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from shapely.geometry import Polygon, mapping

from app.components.space_time_prism import Prism, intersect, speed_bounds_for
from app.config import Settings
from app.models import AnchorPair, GeoEllipse


@dataclass(frozen=True)
class PrismResult:
    prism: Prism
    base_ellipse: GeoEllipse
    mobr_geojson: dict
    base_polygon_geojson: dict


class SpaceTimeReasoner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def compute(self, pair: AnchorPair, domain: str) -> PrismResult:
        bounds = speed_bounds_for(
            domain,
            vessel_v_max_kts=self.settings.vessel_v_max_kts,
            vehicle_v_max_kmh=self.settings.vehicle_v_max_kmh,
        )
        prism = Prism.compute(pair, bounds)
        return PrismResult(
            prism=prism,
            base_ellipse=prism.base_ellipse,
            mobr_geojson=mapping(prism.mobr()),
            base_polygon_geojson=mapping(prism.ellipse_polygon()),
        )

    async def intersect_pairwise(self, prisms: Iterable[Prism]) -> list[Polygon]:
        ps = list(prisms)
        out: list[Polygon] = []
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                inter = intersect(ps[i], ps[j])
                if not inter.is_empty:
                    out.append(inter)
        return out
