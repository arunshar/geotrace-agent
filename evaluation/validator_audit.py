"""Validator stress audit (deterministic, no LLM).

The kinematic validator (`app/agents/validator.py`) gates the OBSERVATIONS: two
consecutive anchors must be mutually reachable under the per-domain max speed,
i.e. it raises `KinematicViolation` when the implied required speed
`dist(a, b) / (t_b - t_a)` exceeds `v_max * 1.05`. It deliberately does NOT gate
on a rendezvous region's bounding-box diagonal (a region is a set of meeting
points, not a path), so this audit stresses the real gate: random anchor pairs.

Reports the detection rate on infeasible pairs (speed > 5% over the envelope) and
the false-positive rate on feasible pairs. Seeded, so it is reproducible.

Writes `evaluation/validator_audit_results/<timestamp>.{md,json}`.
"""

from __future__ import annotations

import asyncio
import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.agents.validator import ValidatorAgent
from app.components.space_time_prism import haversine_m, speed_bounds_for
from app.config import Settings
from app.models import Anchor


async def main(n: int = 200, seed: int = 42, domain: str = "vessel") -> int:
    rnd = random.Random(seed)
    s = Settings()
    val = ValidatorAgent(s)
    bounds = speed_bounds_for(
        domain, vessel_v_max_kts=s.vessel_v_max_kts, vehicle_v_max_kmh=s.vehicle_v_max_kmh
    )
    thr = bounds.v_max_mps * 1.05
    t0 = datetime(2026, 1, 15, 6, 0, 0, tzinfo=UTC)

    tp = tn = fp = fn = 0
    for _ in range(n):
        dt_s = rnd.uniform(600, 21600)
        a_lat, a_lon = rnd.uniform(54.0, 58.0), rnd.uniform(-164.0, -160.0)
        b_lat = a_lat + rnd.uniform(-0.5, 0.5)
        b_lon = a_lon + rnd.uniform(-0.9, 0.9)
        v_req = haversine_m(a_lat, a_lon, b_lat, b_lon) / dt_s
        expected_violation = v_req > thr
        a = Anchor(lat=a_lat, lon=a_lon, t=t0)
        b = Anchor(lat=b_lat, lon=b_lon, t=t0 + timedelta(seconds=dt_s))
        try:
            await val.validate([], domain=domain, anchors=[a, b])
            raised = False
        except Exception:
            raised = True
        if expected_violation and raised:
            tp += 1
        elif expected_violation and not raised:
            fn += 1
        elif (not expected_violation) and raised:
            fp += 1
        else:
            tn += 1

    n_violating = tp + fn
    n_feasible = tn + fp
    detection_rate = tp / n_violating if n_violating else None
    false_positive_rate = fp / n_feasible if n_feasible else None
    summary = {
        "n": n,
        "seed": seed,
        "domain": domain,
        "v_max_mps": bounds.v_max_mps,
        "threshold_mps": thr,
        "n_violating": n_violating,
        "n_feasible": n_feasible,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
    }

    md = "\n".join([
        "# Validator Stress Audit",
        "",
        f"- {n} random {domain} anchor pairs, seed={seed}",
        f"- Per-domain envelope v_max = {bounds.v_max_mps:.3f} m/s; gate fires above v_max x 1.05 = {thr:.3f} m/s",
        f"- Infeasible pairs (required speed > 5% over envelope): {n_violating}",
        f"- Feasible pairs: {n_feasible}",
        "",
        f"**Detection rate on infeasible pairs: {detection_rate:.0%}** ({tp}/{n_violating} raised KinematicViolation).",
        f"**False-positive rate on feasible pairs: {false_positive_rate:.0%}** ({fp}/{n_feasible}).",
    ])

    out_dir = Path("evaluation/validator_audit_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"{ts}.md").write_text(md)
    (out_dir / f"{ts}.json").write_text(json.dumps(summary, indent=2))
    print(md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(main()))
