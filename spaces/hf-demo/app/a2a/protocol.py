"""Agent-to-Agent (A2A) protocol layer.

JSON-RPC 2.0 over HTTP. Each agent advertises a Capability Card at
`/a2a/.well-known/capabilities`. Cards declare:

- name, version
- list of capabilities (e.g., `prism.compute`, `rendezvous.tgard`)
- contact endpoint
- auth scheme (currently bearer, defaults to none in dev)

This file provides:
- `CapabilityCard` schema
- `A2AClient` for outbound calls (used by the orchestrator when a node
  is delegated to a sibling agent in a different process)
- `register_a2a_routes` for the FastAPI server side
"""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)


class CapabilityCard(BaseModel):
    name: str
    version: str
    capabilities: list[str]
    a2a_endpoint: str = Field(..., description="JSON-RPC endpoint URL")
    auth: str = "none"
    contact: str | None = None


class _RpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: dict[str, Any] | None = None
    id: str = Field(default_factory=lambda: uuid4().hex[:12])


class _RpcError(BaseModel):
    code: int
    message: str
    data: Any | None = None


class _RpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: str
    result: Any | None = None
    error: _RpcError | None = None


class A2AClient:
    """Outbound JSON-RPC client. Caches capability cards for 60 s."""

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=2.0))
        self._cards: dict[str, tuple[float, CapabilityCard]] = {}

    async def aclose(self) -> None:
        await self._http.aclose()

    async def card(self, base_url: str) -> CapabilityCard:
        if base_url in self._cards and time.monotonic() - self._cards[base_url][0] < 60.0:
            return self._cards[base_url][1]
        r = await self._http.get(f"{base_url}/a2a/.well-known/capabilities")
        r.raise_for_status()
        card = CapabilityCard.model_validate(r.json())
        self._cards[base_url] = (time.monotonic(), card)
        return card

    async def call(
        self,
        base_url: str,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        trace_id: str | None = None,
    ) -> Any:
        req = _RpcRequest(method=method, params=params or {})
        headers = {"content-type": "application/json"}
        if trace_id:
            headers["traceparent"] = f"00-{trace_id}-{uuid4().hex[:16]}-01"
        r = await self._http.post(f"{base_url}/a2a/jsonrpc", headers=headers, content=req.model_dump_json())
        r.raise_for_status()
        resp = _RpcResponse.model_validate(r.json())
        if resp.error is not None:
            raise RuntimeError(f"A2A {method} failed: {resp.error.code} {resp.error.message}")
        return resp.result


def register_a2a_routes(router: APIRouter, dispatch: dict[str, Any]) -> None:
    @router.post("/a2a/jsonrpc")
    async def jsonrpc(req: Request) -> dict[str, Any]:
        body = await req.body()
        try:
            payload = _RpcRequest.model_validate_json(body)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"bad jsonrpc: {exc}") from exc
        handler = dispatch.get(payload.method)
        if handler is None:
            return _RpcResponse(
                id=payload.id,
                error=_RpcError(code=-32601, message=f"method {payload.method} not found"),
            ).model_dump()
        try:
            result = await handler(payload.params or {})
            return _RpcResponse(id=payload.id, result=result).model_dump()
        except Exception as exc:
            return _RpcResponse(
                id=payload.id,
                error=_RpcError(code=-32000, message=str(exc)),
            ).model_dump()
