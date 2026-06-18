# Contributing to GeoTrace-Agent

Thanks for your interest. This is a research-engineering project; contributions in any of the following areas are welcome:

- New agents (with capability cards in `app/a2a/cards/`).
- New tools registered in `app/agents/tools/__init__.py::REGISTRY`.
- New MCP servers under `app/mcp_servers/`.
- Improvements to the geometric kernel (`app/components/space_time_prism.py`).
- Performance, observability, or documentation improvements.

## Before pushing

Run the local CI gate, which mirrors the GitHub Actions `lint-type-test` job
(ruff as a hard gate, mypy non-blocking, pytest as a hard gate):

```bash
make ci          # or: bash scripts/ci_local.sh
```

On a fresh clone, enable the pre-push hook so the gate runs automatically and
blocks a push when it fails:

```bash
make hooks       # runs: git config core.hooksPath .githooks
```

The gate runs the same `ruff check app observability evaluation tests spaces scripts`
that CI runs, so a lint failure is caught here first. If your local test env is
missing an optional runtime dep (for example `redis` or `rtree`), the gate runs
the collectable test subset and prints a clear `ENV-GAP:` warning rather than
silently passing; CI installs the full dependency set and runs everything.

Contributors who prefer the [pre-commit](https://pre-commit.com) framework can
additionally enable the ruff autofix/format hooks in `.pre-commit-config.yaml`:

```bash
pip install pre-commit && pre-commit install
```

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
