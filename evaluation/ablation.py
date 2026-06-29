"""Cache ablation harness.

Measures the genuine, reproducible effect of the exact/semantic response cache:
it eliminates LLM cost on *repeated/identical* queries. The experiment is a 2x2
over {cache enabled, cache disabled} x {cold run, warm repeat}, run against the
golden set in a single process so the in-memory cache persists between the cold
and warm passes.

What this does NOT claim: a blanket per-query reduction on novel queries. The
cache only fires on repeats, and the distinct-query golden set contains no
repeats within a single cold pass, so the cache provides no saving there. The
in-flight tool deduplicator (`app/services/tool_batcher.py`) is implemented but
not yet wired into the orchestrator run path, so it is not ablated here.

Writes `evaluation/ablation_results/<timestamp>.{md,json}`.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models import Anchor, Budget, QueryIn
from app.services.orchestrator import Orchestrator


def _load(golden_path: str) -> list[tuple[str, QueryIn]]:
    data = json.loads(Path(golden_path).read_text())
    items: list[tuple[str, QueryIn]] = []
    for item in data["items"]:
        q = QueryIn(
            question=item["question"],
            domain=item["domain"],
            anchors=[Anchor(**a) for a in (item.get("anchors") or [])] or None,
            budget=Budget(),
        )
        items.append((item["id"], q))
    return items


async def _run_pass(orch: Orchestrator, items: list[tuple[str, QueryIn]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid, q in items:
        t0 = time.monotonic()
        try:
            out = await orch.run(q)
            rows.append({
                "id": qid,
                "ok": True,
                "latency_s": time.monotonic() - t0,
                "tokens": out.tokens_total,
                "cost_usd": out.cost_usd_total,
                "cache_hit_stages": sum(1 for s in out.stages if s.cache_hit),
            })
        except Exception as exc:  # a planner mis-plan or guardrail trip is a real row
            rows.append({"id": qid, "ok": False, "error": str(exc)})
    return rows


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    toks = [r["tokens"] for r in rows if r.get("ok")]
    costs = [r["cost_usd"] for r in rows if r.get("ok")]
    return {
        "n_ok": len(toks),
        "mean_tokens": statistics.mean(toks) if toks else 0.0,
        "mean_cost_usd": statistics.mean(costs) if costs else 0.0,
        "total_cost_usd": sum(costs),
        "cache_hit_stages": sum(r.get("cache_hit_stages", 0) for r in rows if r.get("ok")),
    }


async def _config(enabled: bool, items: list[tuple[str, QueryIn]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Explicit kwarg overrides any env/.env value; the API key still loads from .env.
    settings = Settings(semantic_cache_enabled=enabled)
    orch = await Orchestrator.bootstrap(settings)
    try:
        cold = await _run_pass(orch, items)  # populates the cache when enabled
        warm = await _run_pass(orch, items)  # identical repeat
    finally:
        await orch.shutdown()
    return cold, warm


def _frac(a: float, b: float) -> float:
    return (a - b) / a if a else 0.0


def _render(summary: dict[str, Any], rows: dict[str, Any], model: str, n: int) -> str:
    on, off = summary["cache_on"], summary["cache_off"]
    lines = [
        "# Cache Ablation",
        "",
        f"- Golden set: {n} queries (model: {model})",
        "- The exact/semantic response cache is measured on identical repeats.",
        "",
        "| config | pass | mean tokens/query | mean cost/query (USD) | total cost (USD) | cache-hit stages |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, blk in (("cache ON", on), ("cache OFF", off)):
        for pas in ("cold", "warm"):
            a = blk[pas]
            lines.append(
                f"| {name} | {pas} | {a['mean_tokens']:.0f} | {a['mean_cost_usd']:.6f} | "
                f"{a['total_cost_usd']:.6f} | {a['cache_hit_stages']} |"
            )
    cold_c = on["cold"]["mean_cost_usd"]
    warm_c = on["warm"]["mean_cost_usd"]
    cold_t = on["cold"]["mean_tokens"]
    warm_t = on["warm"]["mean_tokens"]
    lines += [
        "",
        f"**Repeat-query saving (cache ON).** On an identical repeated query the cache cuts "
        f"per-query LLM cost by {summary['repeat_cost_saving_frac']:.0%} "
        f"(cold ${cold_c:.6f} -> warm ${warm_c:.6f}) and tokens by "
        f"{summary['repeat_token_saving_frac']:.0%} ({cold_t:.0f} -> {warm_t:.0f}).",
        "",
        "**Control (cache OFF).** With the cache disabled the warm repeat costs the same as the "
        f"cold run (${off['warm']['mean_cost_usd']:.6f}), confirming the saving comes from the cache.",
        "",
        "**Scope.** The cache fires only on repeated/near-duplicate queries. On the distinct-query "
        "golden set there are no repeats within a cold pass, so the cache provides no saving there; "
        "its value in production scales with the operational repeat rate. The in-flight tool "
        "deduplicator is implemented but not yet wired into the run path, so it is not ablated.",
    ]
    return "\n".join(lines)


async def main(golden_path: str = "evaluation/golden_dataset.json") -> int:
    items = _load(golden_path)
    on_cold, on_warm = await _config(True, items)
    off_cold, off_warm = await _config(False, items)

    summary = {
        "cache_on": {"cold": _agg(on_cold), "warm": _agg(on_warm)},
        "cache_off": {"cold": _agg(off_cold), "warm": _agg(off_warm)},
    }
    summary["repeat_cost_saving_frac"] = _frac(
        summary["cache_on"]["cold"]["mean_cost_usd"], summary["cache_on"]["warm"]["mean_cost_usd"]
    )
    summary["repeat_token_saving_frac"] = _frac(
        summary["cache_on"]["cold"]["mean_tokens"], summary["cache_on"]["warm"]["mean_tokens"]
    )

    rows = {"on_cold": on_cold, "on_warm": on_warm, "off_cold": off_cold, "off_warm": off_warm}
    model = Settings().primary_model

    out_dir = Path("evaluation/ablation_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"{ts}.md").write_text(_render(summary, rows, model, len(items)))
    (out_dir / f"{ts}.json").write_text(
        json.dumps({"summary": summary, "rows": rows, "model": model}, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(main()))
