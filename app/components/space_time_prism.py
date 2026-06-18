"""Hägerstrand space-time prism kernel.

Given two anchors A = (lat_A, lon_A, t_A) and B = (lat_B, lon_B, t_B)
with t_A < t_B, and a maximum speed v_max, the set of points reachable
from A by t and able to reach B by t_B is, at each interior time t,
a geo-ellipse with foci A and B and semi-major axis

    a(t) = 0.5 * v_max * (t_B - t_A).

The full prism is the union of these ellipses across t in [t_A, t_B].
This module computes prisms, ellipses, MOBRs, and their intersections.

Math is done in a local equirectangular projection centered on the
midpoint of the anchor pair so that distances are Euclidean to first
order. For continental-scale anchors switch to a UTM zone via pyproj.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient as orient_polygon

from app.models import AnchorPair, GeoEllipse, SpeedBounds

EARTH_RADIUS_M = 6_371_000.0
KNOTS_TO_MPS = 0.514_444
KMH_TO_MPS = 1 / 3.6


def speed_bounds_for(domain: str, *, vessel_v_max_kts: float, vehicle_v_max_kmh: float) -> SpeedBounds:
    if domain == "vessel":
        return SpeedBounds(v_max_mps=vessel_v_max_kts * KNOTS_TO_MPS, domain="vessel")
    if domain == "vehicle":
        return SpeedBounds(v_max_mps=vehicle_v_max_kmh * KMH_TO_MPS, domain="vehicle")
    if domain == "pedestrian":
        return SpeedBounds(v_max_mps=2.0, domain="pedestrian")
    if domain == "uav":
        return SpeedBounds(v_max_mps=30.0, domain="uav")
    raise ValueError(f"unknown domain: {domain}")


@dataclass(frozen=True)
class _LocalProjection:
    """Equirectangular projection around a reference latitude."""

    lat_ref_rad: float
    lon_ref_rad: float

    def to_xy(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        x = (lon - self.lon_ref_rad) * math.cos(self.lat_ref_rad) * EARTH_RADIUS_M
        y = (lat - self.lat_ref_rad) * EARTH_RADIUS_M
        return x, y

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        lat = math.degrees(self.lat_ref_rad + y / EARTH_RADIUS_M)
        lon = math.degrees(self.lon_ref_rad + x / (EARTH_RADIUS_M * math.cos(self.lat_ref_rad)))
        return lon, lat


def _projection_for(pair: AnchorPair) -> _LocalProjection:
    return _LocalProjection(
        lat_ref_rad=math.radians(0.5 * (pair.a.lat + pair.b.lat)),
        lon_ref_rad=math.radians(0.5 * (pair.a.lon + pair.b.lon)),
    )


def haversine_m(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    phi1, phi2 = math.radians(a_lat), math.radians(b_lat)
    dphi = phi2 - phi1
    dlam = math.radians(b_lon - a_lon)
    s = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(s)))


@dataclass(frozen=True)
class Prism:
    """A Hägerstrand space-time prism between two anchors.

    `feasible` is False when the anchors are unreachable under v_max,
    in which case `geo_ellipse_at_midpoint` returns a degenerate ellipse.
    """

    pair: AnchorPair
    v_max_mps: float
    duration_s: float
    feasible: bool
    base_ellipse: GeoEllipse
    proj: _LocalProjection

    @classmethod
    def compute(cls, pair: AnchorPair, bounds: SpeedBounds) -> Prism:
        proj = _projection_for(pair)
        ax, ay = proj.to_xy(pair.a.lat, pair.a.lon)
        bx, by = proj.to_xy(pair.b.lat, pair.b.lon)
        d_ab = math.hypot(bx - ax, by - ay)
        duration_s = (pair.b.t - pair.a.t).total_seconds()
        if duration_s <= 0:
            raise ValueError("anchor B must be strictly after anchor A in time")
        L = bounds.v_max_mps * duration_s
        feasible = d_ab - 1e-6 <= L
        # semi-major / semi-minor for the boundary ellipse (the prism's largest cross-section)
        if feasible:
            a_sm = 0.5 * L
            b_sm = 0.5 * math.sqrt(max(0.0, L * L - d_ab * d_ab))
        else:
            a_sm = 0.5 * d_ab
            b_sm = 0.0
        rotation = math.atan2(by - ay, bx - ax)
        return cls(
            pair=pair,
            v_max_mps=bounds.v_max_mps,
            duration_s=duration_s,
            feasible=feasible,
            base_ellipse=GeoEllipse(
                a_lat=pair.a.lat,
                a_lon=pair.a.lon,
                b_lat=pair.b.lat,
                b_lon=pair.b.lon,
                semi_major_m=a_sm,
                semi_minor_m=b_sm,
                rotation_rad=rotation,
            ),
            proj=proj,
        )

    def to_payload(self) -> dict[str, Any]:
        """Flat, JSON-stable serialization for the Temporal activity boundary.

        A Prism holds two live pydantic models (the anchor pair and the base
        ellipse) and a projection dataclass, none of which survive a generic
        dict dump in a form that round-trips through Temporal's data converter.
        This packs the prism into plain JSON so a PRISM activity can hand it to a
        downstream TGARD activity across the boundary; `from_payload` is the exact
        inverse. The round trip is lossless: every field is restored verbatim,
        nothing is recomputed, so the reconstructed prism is identical to the one
        the reasoner produced.
        """

        return {
            "pair": self.pair.model_dump(mode="json"),
            "v_max_mps": self.v_max_mps,
            "duration_s": self.duration_s,
            "feasible": self.feasible,
            "base_ellipse": self.base_ellipse.model_dump(mode="json"),
            "proj": {
                "lat_ref_rad": self.proj.lat_ref_rad,
                "lon_ref_rad": self.proj.lon_ref_rad,
            },
        }

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> Prism:
        """Rebuild a Prism from `to_payload` output (the inverse round trip)."""

        return cls(
            pair=AnchorPair.model_validate(data["pair"]),
            v_max_mps=data["v_max_mps"],
            duration_s=data["duration_s"],
            feasible=data["feasible"],
            base_ellipse=GeoEllipse.model_validate(data["base_ellipse"]),
            proj=_LocalProjection(
                lat_ref_rad=data["proj"]["lat_ref_rad"],
                lon_ref_rad=data["proj"]["lon_ref_rad"],
            ),
        )

    def ellipse_at(self, t: datetime) -> GeoEllipse:
        """Geo-ellipse that contains all (x, y) reachable by time t.

        For an interior time t in (t_A, t_B), the constraint is

            d(p, A) / v_max <= (t - t_A)
            d(p, B) / v_max <= (t_B - t)

        The intersection of the two disks is an algebraic lens that we
        approximate with the inscribed ellipse for fast geometric
        operations. For exact lens geometry use `lens_at`.
        """

        if not (self.pair.a.t <= t <= self.pair.b.t):
            raise ValueError("t is outside prism time interval")
        r_a = self.v_max_mps * (t - self.pair.a.t).total_seconds()
        r_b = self.v_max_mps * (self.pair.b.t - t).total_seconds()
        d_ab = haversine_m(self.pair.a.lat, self.pair.a.lon, self.pair.b.lat, self.pair.b.lon)
        # Inscribed ellipse: foci at A and B, sum-of-distances <= r_a + r_b
        L = max(r_a + r_b, d_ab)
        a_sm = 0.5 * L
        b_sm = 0.5 * math.sqrt(max(0.0, L * L - d_ab * d_ab))
        return GeoEllipse(
            a_lat=self.pair.a.lat,
            a_lon=self.pair.a.lon,
            b_lat=self.pair.b.lat,
            b_lon=self.pair.b.lon,
            semi_major_m=a_sm,
            semi_minor_m=b_sm,
            rotation_rad=self.base_ellipse.rotation_rad,
        )

    def mobr(self, ellipse: GeoEllipse | None = None) -> Polygon:
        """Minimum Orthogonal Bounding Rectangle of an ellipse, in lon/lat."""

        e = ellipse or self.base_ellipse
        # rectangle in local xy aligned with ellipse axes, centered on AB midpoint
        cx = 0.5 * (self.proj.to_xy(self.pair.a.lat, self.pair.a.lon)[0]
                    + self.proj.to_xy(self.pair.b.lat, self.pair.b.lon)[0])
        cy = 0.5 * (self.proj.to_xy(self.pair.a.lat, self.pair.a.lon)[1]
                    + self.proj.to_xy(self.pair.b.lat, self.pair.b.lon)[1])
        a, b, theta = e.semi_major_m, e.semi_minor_m, e.rotation_rad
        local = Polygon([(-a, -b), (a, -b), (a, b), (-a, b)])
        local = rotate(local, math.degrees(theta), origin=(0, 0))
        local = translate(local, cx, cy)
        coords = [self.proj.to_lonlat(x, y) for x, y in local.exterior.coords]
        return orient_polygon(Polygon(coords), sign=1.0)

    def ellipse_polygon(self, ellipse: GeoEllipse | None = None, n: int = 64) -> Polygon:
        e = ellipse or self.base_ellipse
        ax, ay = self.proj.to_xy(self.pair.a.lat, self.pair.a.lon)
        bx, by = self.proj.to_xy(self.pair.b.lat, self.pair.b.lon)
        cx, cy = 0.5 * (ax + bx), 0.5 * (ay + by)
        a, b, theta = e.semi_major_m, e.semi_minor_m, e.rotation_rad
        ts = np.linspace(0, 2 * math.pi, n, endpoint=False)
        xs = cx + a * np.cos(ts) * math.cos(theta) - b * np.sin(ts) * math.sin(theta)
        ys = cy + a * np.cos(ts) * math.sin(theta) + b * np.sin(ts) * math.cos(theta)
        coords = [self.proj.to_lonlat(float(x), float(y)) for x, y in zip(xs, ys, strict=True)]
        return orient_polygon(Polygon(coords), sign=1.0)

    @staticmethod
    def merge_dynamic(ellipses: Iterable[GeoEllipse], pair: AnchorPair) -> Polygon:
        """Dynamic Region Merge: maximal-union of overlapping ellipses.

        Mirrors STAGD-DRM: indexes ellipses by R*-tree and unions
        connected components. We do the trivial all-pairs union here;
        callers needing scale should swap in the rtree.index lookup.
        """

        proj = _projection_for(pair)
        polys: list[Polygon] = []
        for e in ellipses:
            p = Prism(
                pair=pair,
                v_max_mps=0.0,
                duration_s=0.0,
                feasible=True,
                base_ellipse=e,
                proj=proj,
            )
            polys.append(p.ellipse_polygon())
        if not polys:
            return Polygon()
        merged = polys[0]
        for poly in polys[1:]:
            merged = merged.union(poly)
        return merged


def intersect(a: Prism, b: Prism, *, n_slices: int = 16) -> Polygon:
    """Time-slice intersection of two prisms.

    Returns the union of per-slice ellipse intersections within the
    overlapping time window. Empty if the time windows do not overlap.
    """

    t0 = max(a.pair.a.t, b.pair.a.t)
    t1 = min(a.pair.b.t, b.pair.b.t)
    if t0 >= t1:
        return Polygon()
    slices = np.linspace(0.0, 1.0, n_slices)
    accum = Polygon()
    for s in slices:
        t = t0 + (t1 - t0) * float(s)
        ea = a.ellipse_at(t)
        eb = b.ellipse_at(t)
        pa = a.ellipse_polygon(ea)
        pb = b.ellipse_polygon(eb)
        inter = pa.intersection(pb)
        if not inter.is_empty:
            accum = accum.union(inter)
    return accum
