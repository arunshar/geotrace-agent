"""MCP server: AIS history tool.

Trajectory snippets for a vessel ID over a time range. Backed by a
Postgres+PostGIS table populated from MarineCadastre AIS dumps. The
real loader lives in `scripts/seed.py`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import asyncpg

from app.config import get_settings

_TOOL = {
    "ais.history": {
        "description": "Return AIS positions for a vessel ID between two timestamps",
        "inputSchema": {
            "type": "object",
            "required": ["mmsi", "t_start", "t_end"],
            "properties": {
                "mmsi":     {"type": "integer", "minimum": 1, "maximum": 999_999_999},
                "t_start":  {"type": "string", "format": "date-time"},
                "t_end":    {"type": "string", "format": "date-time"},
                "max_rows": {"type": "integer", "minimum": 1, "maximum": 50_000, "default": 5_000},
            },
        },
    },
}


async def _connect() -> asyncpg.Connection:
    s = get_settings()
    return await asyncpg.connect(s.pg_dsn.replace("+asyncpg", ""))


async def history(args: dict[str, Any]) -> list[dict[str, Any]]:
    conn = await _connect()
    try:
        rows = await conn.fetch(
            """
            SELECT mmsi, lat, lon, t, sog_kts, cog_deg
              FROM ais_positions
             WHERE mmsi = $1 AND t >= $2 AND t < $3
          ORDER BY t
             LIMIT $4
            """,
            int(args["mmsi"]),
            datetime.fromisoformat(args["t_start"]),
            datetime.fromisoformat(args["t_end"]),
            int(args.get("max_rows", 5_000)),
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()
