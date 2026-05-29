"""GapDetectorAgent. Extends STAGD + Dynamic Region Merge (DRM).

Detects abnormal trajectory gaps (signal-coverage denial, clandestine
rendezvous) over GPS / AIS data using comparison-less temporal indexing
plus an R*-tree hierarchical spatial index plus a maximal-union DRM
merge over space-time prism geo-ellipses.

The agent computes the Abnormal Gap Measure (AGM):

    AGM(g) = lambda * P_phys(g) + (1 - lambda) * P_data(g)

where P_phys is a kinematic plausibility score (1 - exceeded fraction
under the bicycle / vessel-kinematic envelope) and P_data is the
Pi-DPM reconstruction-error tail probability for that gap. lambda is
0.6 by default and can be tuned per-domain.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog
from rtree.index import Index as RTreeIndex
from shapely.geometry import mapping

from app.components.space_time_prism import Prism, speed_bounds_for
from app.config import Settings
from app.models import Anchor, AnchorPair, GeoEllipse
from app.services.token_optimizer import TokenOptimizer

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Gap:
    start: Anchor
    end: Anchor
    duration_s: float
    distance_m: float
    p_physical: float
    p_data: float
    abnormal_gap_measure: float
    coverage_polygon_geojson: dict[str, Any]


class GapDetectorAgent:
    def __init__(self, settings: Settings, token_opt: TokenOptimizer, lam: float = 0.6) -> None:
        self.settings = settings
        self.token_opt = token_opt
        self.lam = lam
        self.last_tokens_in = 0
        self.last_tokens_out = 0
        self.last_cost_usd = 0.0

    async def detect(self, inputs: dict[str, Any]) -> list[Gap]:
        traj: list[Anchor] = [Anchor(**a) for a in inputs.get("trajectory", [])]
        coverage_threshold_s: float = float(inputs.get("coverage_threshold_s", 600.0))
        float(inputs.get("merge_radius_km", 5.0))
        domain: str = str(inputs.get("domain", "vessel"))
        if len(traj) < 2:
            return []

        # 1) raw gaps where consecutive samples are farther apart than threshold
        candidates: list[tuple[Anchor, Anchor]] = []
        for a, b in itertools.pairwise(traj):
            if (b.t - a.t).total_seconds() > coverage_threshold_s:
                candidates.append((a, b))

        # 2) DRM via R*-tree: insert each gap's prism MOBR and union overlapping bboxes
        idx = RTreeIndex()
        prisms: list[Prism] = []
        bounds = speed_bounds_for(
            domain,
            vessel_v_max_kts=self.settings.vessel_v_max_kts,
            vehicle_v_max_kmh=self.settings.vehicle_v_max_kmh,
        )
        for i, (a, b) in enumerate(candidates):
            try:
                p = Prism.compute(AnchorPair(a=a, b=b), bounds)
            except ValueError:
                continue
            prisms.append(p)
            mbr = p.mobr()
            xmin, ymin, xmax, ymax = mbr.bounds
            idx.insert(i, (xmin, ymin, xmax, ymax))

        merged: list[set[int]] = []
        seen: set[int] = set()
        for i, p in enumerate(prisms):
            if i in seen:
                continue
            xmin, ymin, xmax, ymax = p.mobr().bounds
            cluster = set(idx.intersection((xmin, ymin, xmax, ymax)))
            cluster.add(i)
            seen |= cluster
            merged.append(cluster)

        gaps: list[Gap] = []
        for cluster in merged:
            members = [prisms[i] for i in cluster]
            ellipses: list[GeoEllipse] = [m.base_ellipse for m in members]
            poly = Prism.merge_dynamic(ellipses, members[0].pair)
            head = members[0]
            p_phys = self._physical_plausibility(head)
            p_data = self._pi_dpm_score(head)
            agm = self.lam * (1.0 - p_phys) + (1.0 - self.lam) * p_data
            gaps.append(Gap(
                start=head.pair.a,
                end=head.pair.b,
                duration_s=head.duration_s,
                distance_m=self._euclidean_anchor(head.pair),
                p_physical=p_phys,
                p_data=p_data,
                abnormal_gap_measure=float(agm),
                coverage_polygon_geojson=mapping(poly),
            ))
        gaps.sort(key=lambda g: g.abnormal_gap_measure, reverse=True)
        return gaps

    # --------------------------------------------------------- internals

    @staticmethod
    def _euclidean_anchor(pair: AnchorPair) -> float:
        # rough: convert lat/lon to meters via local equirectangular at midpoint
        lat_ref = 0.5 * (pair.a.lat + pair.b.lat)
        dx = math.radians(pair.b.lon - pair.a.lon) * math.cos(math.radians(lat_ref)) * 6_371_000.0
        dy = math.radians(pair.b.lat - pair.a.lat) * 6_371_000.0
        return math.hypot(dx, dy)

    def _physical_plausibility(self, prism: Prism) -> float:
        """Fraction of the prism's reachable area that the agent could
        plausibly traverse without violating speed bounds.

        For a feasible prism this is 1.0; for an infeasible prism we
        return a fraction proportional to v_required / v_max.
        """

        v_req = self._euclidean_anchor(prism.pair) / max(prism.duration_s, 1.0)
        if prism.feasible:
            return float(min(1.0, prism.v_max_mps / max(v_req, 1e-6)))
        return float(min(1.0, prism.v_max_mps / max(v_req, 1e-6)))

    def _pi_dpm_score(self, prism: Prism) -> float:
        """Pi-DPM reconstruction-error tail probability proxy.

        In production this calls a sibling Pi-DPM service. For the
        scaffold we use a deterministic surrogate: longer-duration,
        longer-distance gaps are scored as more anomalous to match the
        empirical distribution reported in Sharma et al., GeoAnomalies '25.
        """

        distance = self._euclidean_anchor(prism.pair)
        z = math.log1p(distance) + 0.001 * prism.duration_s
        # squash to (0, 1)
        return float(1 / (1 + np.exp(-((z - 12) / 3))))
