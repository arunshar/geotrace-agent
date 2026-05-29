# Contributing to GeoTrace-Agent

Thanks for your interest. This is a research-engineering project; contributions in any of the following areas are welcome:

- New agents (with capability cards in `app/a2a/cards/`).
- New tools registered in `app/agents/tools/__init__.py::REGISTRY`.
- New MCP servers under `app/mcp_servers/`.
- Improvements to the geometric kernel (`app/components/space_time_prism.py`).
- Performance, observability, or documentation improvements.

## Before you open a PR

Run all four locally:

```bash
ruff check .
mypy app
pytest -q
docker compose up --build  # (optional, end-to-end smoke)
```

The PR template (`.github/PULL_REQUEST_TEMPLATE.md`) lists the exact checklist. If you change the geometric kernel, please add a property test in `tests/test_prism.py`.

## Hard invariants

Some invariants are not testable as unit tests; keep them in mind before editing the orchestrator, the validator, or the budget code path. Examples:

- Every region returned to the user MUST go through `ValidatorAgent`.
- All LLM calls go through `app/services/token_optimizer.py::TokenOptimizer`.
- Per-stage spans must include `tool.cache_hit`, `tool.cost_usd`, `tool.tokens_in`, `tool.tokens_out`.

## Versioning prompts

Prompts are versioned by name (e.g. `planner.v3`). Never edit a deployed version in place; bump the version and pin in `prompts.yaml`.

## Reporting issues

See `.github/ISSUE_TEMPLATE/` for templates. Include the `trace_id` from the response (printed in OTEL traces and the `QueryOut.trace_id` field).
