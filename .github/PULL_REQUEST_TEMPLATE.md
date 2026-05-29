# Pull request

## Summary

What changed and why?

## Checklist

- [ ] `ruff check .` is clean.
- [ ] `mypy app` is clean.
- [ ] `pytest -q` is clean.
- [ ] If a new tool was added, it is registered in `app/agents/tools/__init__.py::REGISTRY` and has a capability card under `app/a2a/cards/`.
- [ ] If a prompt was modified, it is a new version (`planner.v4` etc.), not an in-place edit of a deployed version.
- [ ] If the geometric kernel was touched, a property test was added in `tests/test_prism.py`.
- [ ] No new third-party API call without a circuit breaker (`tenacity` + `httpx` timeout).

## Trace examples (optional)

If the change affects user-visible behavior, paste a sample `trace_id` from a successful run.
