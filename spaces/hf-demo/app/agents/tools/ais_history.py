"""AIS history tool. Thin wrapper over `app/mcp_servers/ais_mcp.py::history`."""

from __future__ import annotations

from typing import Any

from app.mcp_servers.ais_mcp import history


async def run(args: dict[str, Any]) -> list[dict[str, Any]]:
    return await history(args)
