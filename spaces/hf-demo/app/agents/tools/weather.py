"""Weather tool. Copernicus CDS in production; stub returns calm seas."""

from __future__ import annotations

from typing import Any


async def run(args: dict[str, Any]) -> dict[str, Any]:
    return {"sea_state": 2, "wind_kts": 8.0, "visibility_nm": 10.0}
