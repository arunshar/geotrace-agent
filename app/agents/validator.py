"""ValidatorAgent. Hard kinematic invariant gate.

Every region returned to the user must pass through this agent. Failure
raises `KinematicViolation`, which the orchestrator surfaces as HTTP 422.
"""

from __future__ import annotations

from shapely.geometry import shape

from app.components.space_time_prism import haversine_m, speed_bounds_for
from app.config import Settings
from app.errors import KinematicViolation
from app.models import RendezvousRegion


class ValidatorAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def validate(
        self,
        regions: list[RendezvousRegion],
        *,
        domain: str = "vessel",
    ) -> list[RendezvousRegion]:
        bounds = speed_bounds_for(
            domain,
            vessel_v_max_kts=self.settings.vessel_v_max_kts,
            vehicle_v_max_kmh=self.settings.vehicle_v_max_kmh,
        )
        out: list[RendezvousRegion] = []
        for r in regions:
            geom = shape(r.polygon_geojson)
            cx, cy = geom.centroid.x, geom.centroid.y
            window_s = (r.latest_meet_t - r.earliest_meet_t).total_seconds()
            if window_s < 0:
                raise KinematicViolation("region time window is reversed", region=r.model_dump(mode="json"))
            v_req_proxy = haversine_m(cy, cx, cy, cx) / max(window_s, 1.0)
            if v_req_proxy > bounds.v_max_mps * 1.05:
                raise KinematicViolation(
                    "region implies infeasible required speed",
                    v_req=v_req_proxy, v_max=bounds.v_max_mps,
                )
            out.append(r)
        return out
