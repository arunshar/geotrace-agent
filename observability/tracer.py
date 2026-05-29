"""OpenTelemetry tracer + a `span` context manager.

Spans propagate to the OTEL collector specified by `GT_OTEL_ENDPOINT`.
Every tool call writes attributes:

- `tool.name`
- `tool.cache_hit`
- `tool.cost_usd`
- `tool.tokens_in`, `tool.tokens_out`

Tail-sampled by Tempo: erroring traces are kept at 100 percent;
successful traces at 1 percent. The collector config lives in
`observability/otel-config.yaml`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config import Settings

_provider: TracerProvider | None = None


def configure_tracing(settings: Settings) -> None:
    global _provider
    if _provider is not None:
        return
    resource = Resource.create({"service.name": "geotrace-agent", "service.version": settings.version})
    _provider = TracerProvider(resource=resource)
    # Only attach an OTLP exporter when an endpoint is configured.
    # Local dev / tests / HF Space / offline runs keep the spans in-memory.
    if settings.otel_endpoint:
        exporter = OTLPSpanExporter(endpoint=f"{settings.otel_endpoint}/v1/traces")
        _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    tracer = trace.get_tracer("geotrace.agent")
    with tracer.start_as_current_span(name) as s:
        if attributes:
            for k, v in attributes.items():
                s.set_attribute(k, v if isinstance(v, (str, int, float, bool)) else str(v))
        yield s
