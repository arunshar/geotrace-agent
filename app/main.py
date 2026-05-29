"""FastAPI entry. Versioned, observability-first, budget-enforced."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import get_settings
from app.errors import GeoTraceError
from app.models import (
    FeedbackIn,
    FeedbackOut,
    HealthOut,
    QueryIn,
    QueryOut,
)
from app.services.orchestrator import Orchestrator
from observability.tracer import configure_tracing, span

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_tracing(settings)
    app.state.orchestrator = await Orchestrator.bootstrap(settings)
    log.info("startup", env=settings.env, version=settings.version)
    try:
        yield
    finally:
        await app.state.orchestrator.shutdown()
        log.info("shutdown")


app = FastAPI(
    title="GeoTrace-Agent",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(GeoTraceError)
async def geotrace_error_handler(_: Request, err: GeoTraceError) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=err.http_status,
        content={"code": err.code, "message": err.message, "context": err.context},
    )


@app.get("/healthz", response_model=HealthOut)
async def healthz() -> HealthOut:
    return HealthOut(status="ok", version=app.version)


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/query", response_model=QueryOut)
async def v1_query(payload: QueryIn, request: Request) -> QueryOut:
    orch: Orchestrator = request.app.state.orchestrator
    with span("api.v1.query", attributes={"q.len": len(payload.question)}):
        try:
            return await orch.run(payload)
        except GeoTraceError:
            raise
        except Exception as exc:  # pragma: no cover
            log.exception("orchestrator_failed", err=str(exc))
            raise HTTPException(status_code=500, detail="internal_error") from exc


@app.post("/v1/feedback", response_model=FeedbackOut)
async def v1_feedback(payload: FeedbackIn, request: Request) -> FeedbackOut:
    orch: Orchestrator = request.app.state.orchestrator
    return await orch.record_feedback(payload)


# Surface the A2A capability card directly so other agents can discover us.
@app.get("/a2a/.well-known/capabilities")
async def capabilities(request: Request) -> dict:
    orch: Orchestrator = request.app.state.orchestrator
    return orch.capability_card()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
