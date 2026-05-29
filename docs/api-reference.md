# API Reference

## `GET /healthz` → `HealthOut`

Liveness probe. Returns 200 with `{"status": "ok", "version": "..."}`.

## `POST /v1/query` → `QueryOut`

Body: `QueryIn`

```json
{
  "question": "Could VESSEL-1234 have rendezvoused with VESSEL-9876 between 06:00Z and 12:00Z near 56N 162W?",
  "domain": "vessel",
  "anchors": [
    {"lat": 56.10, "lon": -162.05, "t": "2026-01-15T06:00:00Z"},
    {"lat": 56.30, "lon": -162.40, "t": "2026-01-15T12:00:00Z"}
  ],
  "budget": {"max_tokens": 12000, "max_tools": 8, "max_seconds": 30}
}
```

Response: `QueryOut` includes `answer`, `regions[]`, `confidence`,
`trace_id`, `stages[]`, `tokens_total`, `cost_usd_total`, and
`hitl_required`. `trace_id` is the OTEL trace identifier.

## `POST /v1/feedback` → `FeedbackOut`

Body: `FeedbackIn`. Records a HITL reviewer's verdict for a previous
trace. Used by offline eval and to seed DPO datasets in `pi-grpo`.

## `GET /a2a/.well-known/capabilities` → `CapabilityCard`

Returns the agent's capability card. See `app/a2a/protocol.py`.

## `POST /a2a/jsonrpc` → JSON-RPC 2.0 response

Inter-agent calls. Methods include `prism.compute`,
`rendezvous.tgard`, `rendezvous.dc_tgard`, `validate.kinematic`.

## Errors

Errors are returned as

```json
{ "code": "geotrace.budget_exceeded", "message": "...", "context": {...} }
```

Stable codes are listed in `app/errors.py`.
