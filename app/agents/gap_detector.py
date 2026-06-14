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
        # Lazily-built, cached real Pi-DPM scorer. None until first use;
        # set to False if torch / the vendored package is unavailable, in
        # which case we permanently fall back to the surrogate.
        self._pidpm: Any = None
        self._torch: Any = None

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

    def _ensure_pidpm(self) -> Any:
        """Build and cache the real vendored Pi-DPM model once.

        Returns the PiDPM instance, or None if torch / the vendored
        package cannot be imported (in which case the caller uses the
        surrogate). The import is lazy and guarded so that GeoTrace-Agent
        stays importable without torch installed.
        """

        if self._pidpm is not None:
            # Either a built model or the False sentinel (unavailable).
            return self._pidpm or None
        try:
            import torch

            from app.components.pidpm.scoring import PiDPM
        except ImportError:
            self._pidpm = False
            return None
        try:
            model = PiDPM()  # small default config (seq_len=24, d_model=128)
            model.eval()
            self._torch = torch
            self._pidpm = model
            return model
        except Exception:  # pragma: no cover - defensive, keep surrogate
            self._pidpm = False
            return None

    def _pi_dpm_score(self, prism: Prism) -> float:
        """Pi-DPM reconstruction-error tail probability.

        When torch and the vendored ``app.components.pidpm`` package are
        importable, this calls the real Pi-DPM: the gap segment is
        synthesized as the straight-line interpolation between the
        prism's two anchors, sampled to ``seq_len`` points in local
        metres, scored by the diffusion + physics anomaly head, and the
        anomaly score is mapped to a (0, 1) tail probability via
        ``1 - exp(-score)``. On ImportError or any failure we fall back
        to the deterministic surrogate below, so the agent stays
        importable and usable without torch.

        Surrogate: longer-duration, longer-distance gaps are scored as
        more anomalous, squashed to (0, 1).
        """

        model = self._ensure_pidpm()
        if model is not None:
            try:
                return self._pi_dpm_real_score(prism, model)
            except Exception:  # pragma: no cover - defensive fallback
                log.warning("pidpm_real_score_failed_fallback_surrogate", exc_info=True)

        distance = self._euclidean_anchor(prism.pair)
        z = math.log1p(distance) + 0.001 * prism.duration_s
        # squash to (0, 1)
        return float(1 / (1 + np.exp(-((z - 12) / 3))))

    def _pi_dpm_real_score(self, prism: Prism, model: Any) -> float:
        """Score the gap with the real Pi-DPM and map to a (0, 1) tail prob."""

        torch = self._torch
        seq_len = int(model.cfg.seq_len)
        # Straight-line interpolation between the two anchors in the prism's
        # local equirectangular projection (metres), sampled to seq_len points.
        ax, ay = prism.proj.to_xy(prism.pair.a.lat, prism.pair.a.lon)
        bx, by = prism.proj.to_xy(prism.pair.b.lat, prism.pair.b.lon)
        s = np.linspace(0.0, 1.0, seq_len)
        xs = ax + (bx - ax) * s
        ys = ay + (by - ay) * s
        seg = np.stack([xs, ys], axis=1).astype(np.float32)  # (seq_len, 2), metres
        x = torch.as_tensor(seg, dtype=torch.float32)[None]   # (1, seq_len, 2)
        # Centre per-sample and scale to model space, matching how the
        # denoiser/dataset were trained (see PiDPM.log_prob).
        x = (x - x.mean(dim=1, keepdim=True)) / float(model.cfg.pos_scale)
        out = model.score(x)
        score = float(np.asarray(out.score).reshape(-1)[0])
        # Map a non-negative anomaly score to a (0, 1) tail probability.
        return float(1.0 - math.exp(-max(0.0, score)))
