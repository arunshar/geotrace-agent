"""Microbenchmarks for the prism kernel.

Measures throughput of the operations on the hot path:

- Prism.compute (anchor pair to feasible prism)
- Prism.ellipse_polygon (polygonization)
- Prism.mobr (bounding rectangle)
- intersect (two-prism time-slice intersection)
"""

from __future__ import annotations

import statistics
import time
from datetime import UTC, datetime, timedelta

from app.components.space_time_prism import Prism, intersect, speed_bounds_for
from app.models import Anchor, AnchorPair

_BOUNDS = speed_bounds_for("vessel", vessel_v_max_kts=25.0, vehicle_v_max_kmh=130.0)
_T0 = datetime(2026, 1, 15, 6, tzinfo=UTC)
_PAIR = AnchorPair(
    a=Anchor(lat=56.10, lon=-162.05, t=_T0),
    b=Anchor(lat=56.30, lon=-162.40, t=_T0 + timedelta(hours=6)),
)
_PAIR2 = AnchorPair(
    a=Anchor(lat=56.15, lon=-162.00, t=_T0),
    b=Anchor(lat=56.25, lon=-162.45, t=_T0 + timedelta(hours=6)),
)


def _bench(name: str, fn, n_warm: int = 50, n_meas: int = 1_000) -> None:
    for _ in range(n_warm):
        fn()
    samples_us: list[float] = []
    for _ in range(n_meas):
        t0 = time.perf_counter_ns()
        fn()
        samples_us.append((time.perf_counter_ns() - t0) / 1_000)
    samples_us.sort()
    p50 = samples_us[len(samples_us) // 2]
    p95 = samples_us[int(len(samples_us) * 0.95)]
    mean = statistics.mean(samples_us)
    print(f"{name:30s}  p50={p50:7.1f} us  p95={p95:7.1f} us  mean={mean:7.1f} us  "
          f"throughput={1e6 / mean:8.0f} ops/s")


def main() -> None:
    print(f"benchmarks (n_meas=1000) -- {_PAIR.a.lat:.2f},{_PAIR.a.lon:.2f} → "
          f"{_PAIR.b.lat:.2f},{_PAIR.b.lon:.2f}, dt=6h")
    _bench("Prism.compute", lambda: Prism.compute(_PAIR, _BOUNDS))
    p = Prism.compute(_PAIR, _BOUNDS)
    p2 = Prism.compute(_PAIR2, _BOUNDS)
    _bench("Prism.ellipse_polygon", lambda: p.ellipse_polygon())
    _bench("Prism.mobr", lambda: p.mobr())
    _bench("intersect (8 slices)", lambda: intersect(p, p2, n_slices=8))
    _bench("intersect (24 slices)", lambda: intersect(p, p2, n_slices=24))


if __name__ == "__main__":
    main()
