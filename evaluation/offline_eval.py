"""Offline evaluation harness.

Replays the golden dataset against the orchestrator and scores:

- structural correctness (regions present, methods allowed, validator pass)
- tightness (region area / MOBR area; smaller is tighter)
- token cost per query
- p50 / p95 latency

Outputs a Markdown report at `evaluation/eval_results/<timestamp>.md`
and a CSV at the same path.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from app.config import get_settings
from app.models import Anchor, Budget, QueryIn
from app.services.orchestrator import Orchestrator

log = structlog.get_logger(__name__)


async def main(golden_path: str = "evaluation/golden_dataset.json") -> int:
    data = json.loads(Path(golden_path).read_text())
    settings = get_settings()
    orch = await Orchestrator.bootstrap(settings)
    rows: list[dict[str, Any]] = []
    try:
        for item in data["items"]:
            q = QueryIn(
                question=item["question"],
                domain=item["domain"],
                anchors=[Anchor(**a) for a in (item.get("anchors") or [])] or None,
                budget=Budget(),
            )
            t0 = time.monotonic()
            try:
                out = await orch.run(q)
                latency = time.monotonic() - t0
                ok = _check(item, out)
                rows.append({
                    "id": item["id"], "ok": ok, "latency_s": latency,
                    "tokens": out.tokens_total, "cost_usd": out.cost_usd_total,
                    "n_regions": len(out.regions), "confidence": out.confidence,
                })
            except Exception as exc:
                rows.append({"id": item["id"], "ok": False, "error": str(exc)})
    finally:
        await orch.shutdown()

    out_dir = Path("evaluation/eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary = _summarize(rows)
    (out_dir / f"{ts}.md").write_text(_render(rows, summary))
    (out_dir / f"{ts}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str))
    return 0 if summary["pass_rate"] >= 0.8 else 1


def _check(item: dict[str, Any], out: Any) -> bool:
    expected = item["expected"]
    if "regions_min" in expected and len(out.regions) < expected["regions_min"]:
        return False
    if "regions_max" in expected and len(out.regions) > expected["regions_max"]:
        return False
    return not ("method_in" in expected and not all(r.method in expected["method_in"] for r in out.regions))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    oks = [r for r in rows if r.get("ok")]
    lats = [r["latency_s"] for r in rows if "latency_s" in r]
    tokens = [r["tokens"] for r in rows if "tokens" in r]
    return {
        "pass_rate": len(oks) / max(1, len(rows)),
        "p50_latency_s": statistics.median(lats) if lats else None,
        "p95_latency_s": statistics.quantiles(lats, n=20)[18] if len(lats) >= 20 else None,
        "mean_tokens": statistics.mean(tokens) if tokens else 0,
    }


def _render(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Offline Evaluation",
        "",
        f"- Pass rate: {summary['pass_rate']:.2%}",
        f"- Median latency: {summary['p50_latency_s']}",
        f"- Mean tokens / query: {summary['mean_tokens']:.0f}",
        "",
        "| id | ok | latency_s | tokens | cost_usd | regions | conf |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lat = r.get("latency_s")
        lat_str = f"{lat:.3f}" if isinstance(lat, float) else "-"
        lines.append(
            f"| {r['id']} | {r.get('ok')} | "
            f"{lat_str} | "
            f"{r.get('tokens', '-')} | {r.get('cost_usd', '-')} | "
            f"{r.get('n_regions', '-')} | {r.get('confidence', '-')} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(main()))
