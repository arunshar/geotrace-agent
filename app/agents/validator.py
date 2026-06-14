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
            window_s = (r.latest_meet_t - r.earliest_meet_t).total_seconds()
            if window_s < 0:
                raise KinematicViolation("region time window is reversed", region=r.model_dump(mode="json"))
            # Worst-case required speed across the region: the farthest two
            # corners of the region's lon/lat bounding box must be traversable
            # within the meet window. The previous implementation took the
            # haversine of the centroid against itself (always 0 m), so the
            # speed gate never fired; here we use the bounding-box diagonal as
            # the maximum displacement the moving object would have to cover.
            minx, miny, maxx, maxy = geom.bounds
            span_m = haversine_m(miny, minx, maxy, maxx)
            v_req_proxy = span_m / max(window_s, 1.0)
            if v_req_proxy > bounds.v_max_mps * 1.05:
                raise KinematicViolation(
                    "region implies infeasible required speed",
                    v_req=v_req_proxy, v_max=bounds.v_max_mps,
                )
            out.append(r)
        return out
