# GeoTrace-Agent: Architecture

## Pipeline

```
                        ┌──────────────────────────┐
   POST /v1/query  ───► │     InputGuard (sec)     │
                        └────────────┬─────────────┘
                                     ▼
                        ┌──────────────────────────┐
                        │   QueryRouter / Rewriter │
                        └────────────┬─────────────┘
                                     ▼
                        ┌──────────────────────────┐
                        │     PlannerAgent (LLM)   │  ─── PlanGraph (typed DAG)
                        └────────────┬─────────────┘
                                     ▼
            ┌───────────────────────────────────────────────────┐
            │             Orchestrator (topo-sort)              │
            │   parallel layers; budget gate; OTEL spans         │
            └────────────┬────────────────────────┬─────────────┘
                         ▼                        ▼
        ┌────────────────────────┐  ┌────────────────────────┐
        │ SpaceTimeReasoner      │  │ GapDetectorAgent       │
        │ (Prism / GeoEllipse /  │  │ (STAGD + DRM + Pi-DPM  │
        │  MOBR / intersect)     │  │  AGM scoring)          │
        └────────────┬───────────┘  └────────────┬───────────┘
                     ▼                           ▼
                  ┌────────────────────────────────────┐
                  │  RendezvousFinderAgent             │
                  │  (TGARD, DC-TGARD; bi-directional  │
                  │   pruning; ellipse symmetry)       │
                  └────────────┬───────────────────────┘
                               ▼
                  ┌────────────────────────────────────┐
                  │  ValidatorAgent (kinematic gate)   │
                  └────────────┬───────────────────────┘
                               ▼
                  ┌────────────────────────────────────┐
                  │  TokenOptimizer.summarize → Answer │
                  └────────────────────────────────────┘
```

## Why this shape

The system separates four orthogonal concerns:

1. Geometric truth. `app/components/space_time_prism.py` is deterministic
   and unit-testable. It owns Hägerstrand math, geo-ellipses, MOBRs, and
   DRM unions.
2. Semantic reasoning. The planner is the only LLM-dependent stage that
   gets to plan; the other agents either run deterministic kernels or
   call the planner's-permitted tools.
3. Budgets. Every LLM call goes through `TokenOptimizer.call_llm_*`. The
   orchestrator stops on token / tool / wallclock overrun and returns
   `terminated_by_budget=true` so callers can surface a partial answer.
4. Provenance. Every stage emits an OTEL span with `tool.cost_usd`,
   `tool.tokens_*`, `tool.cache_hit`, and the trace ID flows out to the
   user in `QueryOut.trace_id` so analysts can drill in.

## Token-consumption optimization

| Mechanism | Saving |
|---|---|
| Adaptive prompt compression | drops ~50% of tokens for >2k-token plans |
| Prefix-cache–aware system prompt | exploits Anthropic prompt caching (cache_control: ephemeral) |
| Structured outputs | one-shot retry with delta correction; eliminates parsing-failure re-runs |
| Per-call `max_tokens` clamp | run budget never overshoots even on planner regressions |
| Semantic cache (exact + near) | identical questions return without an LLM call |
| In-flight tool dedup | parallel agents requesting the same tool share one awaitable |

## Tool-call optimization

- Tool batcher (`app/services/tool_batcher.py`) collapses concurrent
  identical calls into a single execution.
- Semantic cache (`app/services/semantic_cache.py`) stores per-call
  outputs by `(tool, args)` and surfaces near-hits via embeddings.
- The planner emits parallel-safe layers (`PlanGraph.topo_layers`); the
  orchestrator spawns them with `asyncio.gather`.

## Chain of thought, made auditable

The planner's reasoning is not a free-form transcript: it is the
`PlanGraph` plus the per-node `rationale` and `confidence_prior`. Every
plan is stored in Postgres alongside its `trace_id`. Replays are exact
because the planner prompt is versioned (`planner.v3`) and the LLM call
is cached.

## SOTA agent protocols

- MCP. `app/mcp_servers/prism_mcp.py` exposes the prism kernel as a
  tool over JSON-RPC stdio so any MCP-aware client can call it.
- A2A. `app/a2a/protocol.py` registers JSON-RPC endpoints under
  `/a2a/jsonrpc` with capability cards at
  `/a2a/.well-known/capabilities`. Outbound calls cache cards for 60 s.

## Human-in-the-loop

Items with confidence < 0.7 (configurable) are pushed to a Postgres
HITL queue and reviewed in the Streamlit ops console. The reviewer's
verdict is captured by `observability/feedback.py` and can be exported
as a preference dataset for the sibling `pi-grpo` project (DPO).
