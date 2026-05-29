"""Road network tool. Returns travel-time bounds for a bbox.

In production: queries OSRM or Valhalla against an OSM extract. The
returned bounds tighten the prism's effective speed inside cities.
"""

from __future__ import annotations

from typing import Any


async def run(args: dict[str, Any]) -> dict[str, Any]:
    bbox = args.get("bbox") or {}
    return {"bbox": bbox, "v_max_kmh_effective": 80.0, "median_speed_kmh": 35.0}
