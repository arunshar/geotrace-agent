"""Geometric invariants for the space-time prism kernel."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from app.components.space_time_prism import Prism, haversine_m, intersect, speed_bounds_for
from app.models import Anchor, AnchorPair


def _pair(t0: datetime, t1: datetime, *, lat=56.0, lon=-162.0, dlat=0.2, dlon=-0.4) -> AnchorPair:
    return AnchorPair(
        a=Anchor(lat=lat, lon=lon, t=t0),
        b=Anchor(lat=lat + dlat, lon=lon + dlon, t=t1),
    )


def test_feasible_prism_has_valid_ellipse() -> None:
    bounds = speed_bounds_for("vessel", vessel_v_max_kts=25.0, vehicle_v_max_kmh=130.0)
    t0 = datetime(2026, 1, 15, 6, tzinfo=UTC)
    pair = _pair(t0, t0 + timedelta(hours=6))
    p = Prism.compute(pair, bounds)
    assert p.feasible
    e = p.base_ellipse
    assert e.semi_major_m >= e.semi_minor_m >= 0
    assert e.semi_major_m > 0


def test_infeasible_when_distance_exceeds_speed_budget() -> None:
    bounds = speed_bounds_for("vessel", vessel_v_max_kts=2.0, vehicle_v_max_kmh=130.0)
    t0 = datetime(2026, 1, 15, 6, tzinfo=UTC)
    # 1000 km in 1 hour at 2 kts is hopeless
    pair = AnchorPair(
        a=Anchor(lat=0.0, lon=0.0, t=t0),
        b=Anchor(lat=0.0, lon=10.0, t=t0 + timedelta(hours=1)),
    )
    p = Prism.compute(pair, bounds)
    assert not p.feasible
    assert p.base_ellipse.semi_minor_m == 0.0


def test_intersection_empty_when_time_windows_dont_overlap() -> None:
    bounds = speed_bounds_for("vessel", vessel_v_max_kts=25.0, vehicle_v_max_kmh=130.0)
    t0 = datetime(2026, 1, 15, 6, tzinfo=UTC)
    a = Prism.compute(_pair(t0, t0 + timedelta(hours=2)), bounds)
    b = Prism.compute(_pair(t0 + timedelta(hours=4), t0 + timedelta(hours=6)), bounds)
    inter = intersect(a, b, n_slices=8)
    assert inter.is_empty


def test_haversine_symmetric_and_zero_on_self() -> None:
    assert haversine_m(0, 0, 0, 0) == 0.0
    d1 = haversine_m(40.0, -70.0, 41.0, -71.0)
    d2 = haversine_m(41.0, -71.0, 40.0, -70.0)
    assert math.isclose(d1, d2, rel_tol=1e-9)


def test_anchor_b_must_follow_anchor_a_in_time() -> None:
    bounds = speed_bounds_for("vessel", vessel_v_max_kts=25.0, vehicle_v_max_kmh=130.0)
    t0 = datetime(2026, 1, 15, 6, tzinfo=UTC)
    pair = AnchorPair(
        a=Anchor(lat=0.0, lon=0.0, t=t0 + timedelta(hours=1)),
        b=Anchor(lat=0.0, lon=0.1, t=t0),
    )
    with pytest.raises(ValueError):
        Prism.compute(pair, bounds)
