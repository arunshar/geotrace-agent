"""MCP server: space-time prism tool.

Exposes the prism kernel over the Model Context Protocol so any
MCP-aware client, IDE plugin, or sibling agent can call
`prism.compute`, `prism.intersect`, `prism.merge_dynamic` without going
through this app's HTTP surface.

The wire format is JSON-RPC 2.0 over stdio (MCP default transport).
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import structlog

from app.components.space_time_prism import (
    Prism,
    speed_bounds_for,
)
from app.config import get_settings
from app.models import Anchor, AnchorPair

log = structlog.get_logger("prism_mcp")


_TOOLS: dict[str, dict[str, Any]] = {
    "prism.compute": {
        "description": "Compute a Hägerstrand space-time prism between two anchors",
        "inputSchema": {
            "type": "object",
            "required": ["anchor_a", "anchor_b", "domain"],
            "properties": {
                "anchor_a": {"$ref": "#/$defs/Anchor"},
                "anchor_b": {"$ref": "#/$defs/Anchor"},
                "domain": {"enum": ["vessel", "vehicle", "pedestrian", "uav"]},
            },
            "$defs": {"Anchor": {
                "type": "object", "required": ["lat", "lon", "t"],
                "properties": {
                    "lat": {"type": "number", "minimum": -90, "maximum": 90},
                    "lon": {"type": "number", "minimum": -180, "maximum": 180},
                    "t":   {"type": "string", "format": "date-time"},
                },
            }},
        },
    },
    "prism.intersect": {
        "description": "Time-slice intersection of two prisms",
        "inputSchema": {"type": "object"},
    },
    "prism.merge_dynamic": {
        "description": "Maximal-union DRM merge of overlapping prism ellipses",
        "inputSchema": {"type": "object"},
    },
}


# ------------------------------------------------------------ JSON-RPC


def _resp(id_: Any, *, result: Any | None = None, error: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {"jsonrpc": "2.0", "id": id_}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    return out


def _initialize(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocolVersion": "2025-03-26",
        "serverInfo": {"name": "geotrace-prism-mcp", "version": "0.1.0"},
        "capabilities": {"tools": {"listChanged": False}},
    }


def _tools_list(_: dict[str, Any]) -> dict[str, Any]:
    return {"tools": [{"name": k, **v} for k, v in _TOOLS.items()]}


async def _tools_call(params: dict[str, Any]) -> dict[str, Any]:
    name = params["name"]
    args = params.get("arguments") or {}
    if name == "prism.compute":
        a = Anchor(**args["anchor_a"])
        b = Anchor(**args["anchor_b"])
        bounds = speed_bounds_for(
            args["domain"],
            vessel_v_max_kts=get_settings().vessel_v_max_kts,
            vehicle_v_max_kmh=get_settings().vehicle_v_max_kmh,
        )
        prism = Prism.compute(AnchorPair(a=a, b=b), bounds)
        from shapely.geometry import mapping
        return {"content": [{
            "type": "text",
            "text": json.dumps({
                "feasible": prism.feasible,
                "duration_s": prism.duration_s,
                "v_max_mps": prism.v_max_mps,
                "ellipse": prism.base_ellipse.model_dump(),
                "polygon_geojson": mapping(prism.ellipse_polygon()),
                "mobr_geojson": mapping(prism.mobr()),
            }, default=str),
        }]}
    if name == "prism.intersect":
        # left as exercise: would deserialize two prisms from cache ids
        return {"content": [{"type": "text", "text": json.dumps({"empty": True})}]}
    if name == "prism.merge_dynamic":
        return {"content": [{"type": "text", "text": json.dumps({"empty": True})}]}
    return {"content": [{"type": "text", "text": f"unknown tool {name}"}], "isError": True}


_HANDLERS: dict[str, Any] = {
    "initialize": _initialize,
    "tools/list": _tools_list,
    "tools/call": _tools_call,
}


async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while True:
        line = await reader.readline()
        if not line:
            return
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = msg.get("method")
        handler = _HANDLERS.get(method)
        if handler is None:
            resp = _resp(msg.get("id"), error={"code": -32601, "message": f"method {method} not found"})
        else:
            try:
                result = handler(msg.get("params") or {})
                if asyncio.iscoroutine(result):
                    result = await result
                resp = _resp(msg.get("id"), result=result)
            except Exception as exc:  # pragma: no cover
                resp = _resp(msg.get("id"), error={"code": -32000, "message": str(exc)})
        writer.write(json.dumps(resp).encode() + b"\n")
        await writer.drain()


async def _stdio_main() -> None:  # pragma: no cover
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    loop = asyncio.get_running_loop()
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    transport, _ = await loop.connect_write_pipe(asyncio.streams.FlowControlMixin, sys.stdout)
    writer = asyncio.StreamWriter(transport, _, reader, loop)
    await serve(reader, writer)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_stdio_main())
