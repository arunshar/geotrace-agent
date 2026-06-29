# GeoTrace-Agent, Explained

> A plain-language tour of what the system actually does and, more importantly, **why each design decision was made**. Written to be read end to end, in the spirit of "here is the one idea, here is the analogy, here is the rationale." Companion to the [project website](https://arunshar.com) and the [full paper](https://arunshar.com/projects/geotrace-agent/).

---

## 0. The one mental model

GeoTrace-Agent answers natural-language questions about moving objects ("could these two vessels have met near here, in this window?"). The single idea that organizes everything:

> **The LLM plans. Deterministic kernels compute. A validator gates. The LLM never decides anything that has a ground truth.**

Think of it as **a compiler and a type system wrapped around a language model.** The LLM writes a program (a plan); the runtime type-checks that program, runs the parts that are exact math with exact math, and refuses to return any answer that is physically impossible. The LLM is used only where judgment is genuinely needed (deciding the steps, narrating the result), not where the answer is decidable by geometry.

Everything below is downstream of that one decision.

---

## 1. The Temporal-style analogy

If durable execution is the one concept that makes Temporal "click," here is the equivalent table for GeoTrace-Agent. Each row is a familiar systems idea mapped onto an agent concept.

| Systems concept you know | GeoTrace-Agent equivalent | Why the analogy holds |
|---|---|---|
| A compile-checked workflow DAG | The **typed PlanGraph** | The plan is a DAG of typed nodes validated against a schema before anything runs; cycles and over-budget plans are rejected at "compile time." |
| A workflow engine running tasks in parallel | The **Orchestrator** | It topo-sorts the DAG and runs each independent layer concurrently with `asyncio.gather`, under a hard token / tool / wallclock budget. |
| A type system (rejects ill-typed programs) | The **kinematic validator** | It rejects any state that violates the physics envelope, raising `KinematicViolation` rather than letting an impossible answer through. |
| gRPC / typed RPC between services | **MCP** (tools) + **A2A** (agents) | The geometry kernel is an MCP server; the agent advertises a capability card and accepts JSON-RPC Agent-to-Agent calls. |
| Distributed tracing (spans, sampling) | **OpenTelemetry + the cost ledger** | Every stage emits a span with token / cost / cache-hit attributes; traces are tail-sampled; a per-stage ledger answers "what did this query cost?" |
| A feedback loop into model training | **HITL queue to Pi-GRPO** | Low-confidence traces go to a human-review queue whose verdicts export as preference pairs for DPO / GRPO post-training. |
| Durable activity boundaries (serialize state across steps) | `Prism.to_payload` / `from_payload` | The prism serializes losslessly so a PRISM step can hand its result to a downstream TGARD step across a Temporal activity boundary. |

The single sentence to remember: **GeoTrace-Agent treats an LLM agent like a distributed program, with a type system (physics), a scheduler (the orchestrator), typed interfaces (MCP / A2A), and full observability, instead of treating it like a chatbot you hope gets it right.**

---

## 2. How the agents actually interact

This is grounded in `app/services/orchestrator.py`. One request flows through `run()`:

```
run(q):
  guard.check(question)                      # input safety
  plan   = planner.plan(q)                   # LLM -> typed PlanGraph
  results = execute_plan(plan)               # the interesting part
  answer = summarize(plan, results)          # LLM narrates the computed facts
  regions = collect_regions(results)
  confidence = aggregate_confidence(...)
  out_filter.scrub(answer)                   # output safety
  if confidence < threshold: hitl.enqueue(...)   # human review
  return QueryOut(answer, regions, trace_id, tokens, cost, hitl_required)
```

`execute_plan` is where the multi-agent coordination lives:

```python
for layer in plan.topo_layers():                       # dependency layers
    tasks = [execute_node(node, ...) for node in layer]
    outs  = await asyncio.gather(*tasks, return_exceptions=True)  # parallel
    for node, out in zip(layer, outs):
        if isinstance(out, KinematicViolation): raise out  # hard gate, fail loud
        results[node.id] = out
    self._enforce_budget(q.budget, ctx)                # stop if over budget
```

Read that closely, it encodes several decisions at once:

- **Topo layers + `asyncio.gather`**: nodes with no remaining dependencies run *concurrently*. Computing two prisms is two parallel tasks, not two sequential calls.
- **`return_exceptions=True`**: one node failing does not crash the run; the orchestrator records the error and continues, except for `KinematicViolation`, which is re-raised immediately (an impossible answer must abort, not degrade).
- **Budget enforced after every layer**: token, tool-call, and wallclock budgets are checked between layers; an overrun sets `terminated_by_budget` and raises `BudgetExceeded`. The agent can never run away.
- **Per-stage ledger**: each stage runs inside `_StageContextManager`, which opens an OpenTelemetry span and appends a `StageTrace` (tokens in/out, cost, cache-hit). That ledger is how cost is attributed per query.

`execute_node` is a dispatch table on the node's typed `kind`: `PRISM` to the SpaceTimeReasoner, `GAPS` to the GapDetector, `TGARD/DC_TGARD` to the RendezvousFinder, `VALIDATE` to the ValidatorAgent, `SUMMARIZE` handled at the end. Each agent has a typed input and a typed output, so the orchestrator never parses free text to decide what to do next.

---

## 3. The design decisions and the rationale (the heart of it)

Each decision below is stated as **what, why, and what the alternative would have cost.**

### 3.1 A typed PlanGraph instead of free-form chain-of-thought
**What.** The planner (`app/agents/planner.py`) does not emit prose. It returns a JSON object validated against a strict schema: a DAG of nodes, each one of `prism.compute`, `gaps.detect`, `rendezvous.tgard`, `rendezvous.dc_tgard`, `validate.kinematic`, `retrieve.semantic`, `summarize`, with `deps`, `inputs`, `expected_tokens`, and a `confidence_prior`.
**Why.** Four properties fall out for free: (a) **auditable**, every reasoning step is a typed record you can inspect; (b) **replayable**, the plan plus the prompt version reproduce the run exactly; (c) **parallelizable**, a DAG has topo layers, so independent steps run concurrently; (d) **bounded**, `sum(expected_tokens)` is checked against the budget *before* any work runs (`PlanInfeasible` otherwise). It also makes caching effective, because a typed plan has stable keys.
**Alternative cost.** Free-form CoT is a string you cannot statically check, cannot safely parallelize, and cannot bound. You would be parsing prose to decide control flow, the thing that makes agents flaky.

### 3.2 Deterministic kernel before the LLM
**What.** Every spatially-decidable sub-problem is computed by a numerical kernel (`app/components/space_time_prism.py`), not the LLM. The space-time prism, the geo-ellipse, the bounding rectangle, the prism intersection, all exact.
**Why.** LLMs hallucinate distances and skip the physics check. Reachability has a *ground truth*: given two anchors and a max speed, the set of reachable points is a geo-ellipse with semi-major axis `a = 0.5 * v_max * duration`. There is no reason to ask a model to approximate something you can compute exactly, reproducibly, and cheaply.
**Alternative cost.** Trusting the LLM for geometry means occasional confident-but-wrong answers, which in maritime safety or homeland security is unacceptable.

### 3.3 A single `TokenOptimizer` choke-point
**What.** Every LLM call in the system goes through one class (`app/services/token_optimizer.py`). It owns five jobs: prompt compression, prefix-cache-aware assembly (`cache_control: ephemeral`), structured-output enforcement with one delta-correction retry, a per-call `max_tokens` clamp against the remaining budget, and cost accounting (returns `tokens_in, tokens_out, cost_usd, cache_hit`).
**Why.** Cross-cutting concerns belong in one place. Budget, caching, retries, timeouts, and cost attribution are *uniform* because there is exactly one door every call walks through. The orchestrator can never overshoot the budget on a planner regression, because the clamp lives at the door.
**Alternative cost.** Scattering LLM calls means scattering budget logic, cost accounting, and retry policy, which drift apart and leak spend.

### 3.4 The efficiency stack
**What.** (1) **Adaptive prompt compression**, prompts over a threshold are head-tail truncated with an elision marker. (2) **Hybrid semantic cache**, an exact-key lookup layered over a near-key embedding lookup, so identical anchor pairs return in O(1) and near-duplicate questions reuse prior answers. (3) **In-flight tool deduplication**, the `ToolBatcher` keys concurrent calls by `(tool, sha256(args))` and collapses duplicates into one awaitable; it is implemented but not yet wired into the orchestrator run path, so it does not contribute to the measured numbers.
**Why.** The two active layers are complementary: the cache eliminates LLM cost on repeated or near-duplicate queries, and compression shrinks each call. On a live golden-set run with Claude Sonnet 4.6 the full pipeline costs about 0.026 USD per query (~2,736 tokens), and a repeated query drops to zero LLM cost on a cache hit (cold to warm; see `evaluation/eval_results/` and `evaluation/ablation_results/`). The cache does not fire on novel queries, so there is no blanket per-query reduction.
**Alternative cost.** Any one layer alone leaves money on the table; the dedup-vs-cache distinction (in-run duplicates vs cross-run repeats) is the subtle part most systems miss.

### 3.5 The kinematic validator as a hard gate (and a subtle rationale)
**What.** `app/agents/validator.py` is the last thing every region passes through. It enforces the Hagerstrand feasibility condition on the **observations**: consecutive anchors must be mutually reachable, `dist(A, B) <= v_max * (t_B - t_A)` (with a 5% tolerance). Failure raises `KinematicViolation`, surfaced as HTTP 422.
**Why this exact form.** The code comment captures a real lesson: an earlier version gated on the rendezvous region's **bounding-box diagonal divided by the meet window** as a "required speed." That was wrong, a region is the *set of alternative meeting points*, not a path to traverse, and its extent scales with the prism duration while the meet window is shorter, so that proxy rejected every valid region. The fix gates the physics where the physics actually lives (the observations), not on a derived artifact.
**Why a hard gate, not a soft penalty.** Physical feasibility is an invariant, not a preference. A soft score could be overridden by a confident summary; a hard gate cannot. The system would rather return nothing than return an impossible region.
**Alternative cost.** A soft penalty, or no gate, lets physically impossible answers reach users, the failure mode that erodes trust fastest.

### 3.6 MCP for tools, A2A for agents
**What.** The prism kernel is exposed as a Model Context Protocol server (`prism.compute`, `prism.intersect`, `prism.merge_dynamic`); the orchestrator advertises a capability card and accepts JSON-RPC 2.0 Agent-to-Agent calls.
**Why.** The geometry kernel is genuinely reusable; making it an MCP server means any MCP-aware client, IDE plugin, or sibling agent can call it without going through this app. A2A makes GeoTrace-Agent a first-class citizen other agents can invoke. This is the same reason you put a typed RPC interface on a useful service.
**Alternative cost.** Burying the kernel behind one app's HTTP surface makes it un-reusable and un-composable with the broader agent ecosystem.

### 3.7 OpenTelemetry plus a per-stage cost ledger
**What.** `observability/tracer.py` wraps every tool call in a span carrying `tool.name, cache_hit, cost_usd, tokens_in/out`; traces are **tail-sampled** (100% of errors, 1% of successes). `observability/cost_tracker.py` writes one row per `(trace_id, stage)`.
**Why.** You cannot optimize what you cannot measure. Per-stage attribution is what made the efficiency work legible (the reproduced per-query cost and the cache cold-to-warm saving came from this ledger, not a guess). Tail sampling keeps every failure for debugging while bounding storage, the right default for an eval stack.
**Alternative cost.** Aggregate metrics tell you the system is slow or expensive but not *where*; you cannot do targeted optimization or honest cost claims without per-stage traces.

### 3.8 Offline golden-set CI gate plus online drift monitor
**What.** Offline (`evaluation/offline_eval.py`): replay a labeled golden set, score structure, tool-trajectory (which algorithm), feasibility, tightness, tokens, latency, and **fail CI if pass-rate < 0.8**. Online (`evaluation/online_monitor.py`): sample 1% of traffic, recompute the validator independently, re-run a frozen-prompt plan and alert on **plan drift**.
**Why.** Evaluation is a discipline, not an afterthought. The CI gate means a prompt or model change that regresses the agent fails the build. The drift monitor catches the silent case, a new model quietly changes the plan distribution on live traffic.
**Alternative cost.** Without a gate, regressions ship; without a drift monitor, you find out from users.

### 3.9 A deterministic feasibility oracle instead of an LLM judge
**What.** The core evaluator is the prism feasibility oracle, not an LLM-as-judge. A claimed region is valid only if it sits inside the physically reachable prism.
**Why.** The domain has ground truth, so a deterministic check is reproducible, has zero judge variance, and is cheap enough to run on 100% of outputs as a guardrail. An LLM judge is itself a model that drifts across versions and needs its own calibration.
**Honest limit.** This covers the structured, checkable parts. It does **not** judge the free-text summary's quality or faithfulness; that is exactly where a calibrated LLM judge belongs, and it is the open piece.

### 3.10 HITL queue that closes the loop to RL
**What.** Traces below a confidence threshold flow to a Postgres human-in-the-loop queue; reviewer verdicts export as preference triples for the sibling **Pi-GRPO** project (DPO / GRPO post-training).
**Why.** It connects agentic reasoning to reward-modeled fine-tuning without changing the LLM call surface or the validator. Human judgment becomes training signal.
**Alternative cost.** A dead-end review queue collects labels nobody uses; wiring it to Pi-GRPO turns review into model improvement.

---

## 4. Core-file walkthrough (the ~8 files that matter)

- **`app/services/orchestrator.py`** , the conductor. `run()` is the request lifecycle; `_execute_plan` is the topo-layer parallel loop; `_enforce_budget` is the governor; `_StageContextManager` is the span + ledger. Start here.
- **`app/agents/planner.py`** , the only place the LLM decides control flow. Emits the typed PlanGraph against `_PLAN_JSON_SCHEMA`; `_coerce_plan` validates the DAG, attaches anchors to `prism.compute` nodes, runs the cycle check, and enforces the token budget.
- **`app/models.py`** , the type system: `PlanGraph` / `PlanNode` / `PlanNodeKind`, `topo_layers()`, `Budget`, `RendezvousRegion`, `GeoEllipse`. These types are the contract every agent honors.
- **`app/components/space_time_prism.py`** , the geometry IP. `Prism.compute` (semi-major `0.5 * v_max * duration`, feasibility `d_ab <= L`), `ellipse_at(t)`, `mobr`, `ellipse_polygon`, `merge_dynamic` (DRM), and `intersect` (time-slice prism intersection, the operation TGARD rides on). Note `to_payload`/`from_payload`, lossless serialization for durable (Temporal) activity boundaries.
- **`app/services/token_optimizer.py`** , the single LLM door (section 3.3). Also holds the offline stub that returns a deterministic plan when no API key is set, which is why the demo runs without a key.
- **`app/agents/validator.py`** , the hard kinematic gate (section 3.5); short, but the rationale comment is the most instructive in the codebase.
- **`evaluation/offline_eval.py` + `evaluation/online_monitor.py`** , the two eval surfaces (section 3.8).
- **`observability/tracer.py` + `observability/cost_tracker.py`** , spans, tail sampling, and the per-stage cost ledger (section 3.7).
- **`spaces/hf-demo/streamlit_app.py`** , the demo (section 5).

The other agents (`space_time_reasoner.py`, `gap_detector.py`, `rendezvous_finder.py`) are thin, typed clients over the prism kernel; the gap detector additionally fuses a Pi-DPM anomaly term into the Abnormal Gap Measure when torch is available, and falls back to a deterministic surrogate otherwise.

---

## 5. The demo, and how to read it

The Hugging Face Space (`spaces/hf-demo/streamlit_app.py`) is the system in miniature, CPU-only, no database. **Sidebar:** pick a preset (rendezvous / gap audit / prism-only), edit the question, the vessel anchors (lat, lon, time), and a budget; click Run. **Main panel, four views of the same run:**

1. **Chain of thought (typed PlanGraph)** , the plan rendered as a flowchart. This is the agent's "program."
2. **Map** , the space-time prism ellipses, their bounding rectangles, and (in green) the candidate rendezvous region. This is the geometry the kernel computed.
3. **Validator** , regions returned, methods used, and `kinematic_pass`. This is the gate's verdict.
4. **Cost / tokens** and the raw **Result JSON** , what the run spent and produced.

It felt opaque before because it shows the *machinery* (plan + map + JSON), not a narrated paragraph. Without an API key it runs the deterministic stub, so the geometry always renders even with no LLM. The fix, if you want it to read more easily, is a one-line "what just happened" summary panel.

---

## 6. The full paper, section by section

The README is the NeurIPS-style paper. Map of where each idea is detailed:

- Section 1 (Introduction) , the deterministic-first thesis and the typed-agent contributions.
- Section 3 (System Architecture) , the request path and the four separated concerns (geometric truth, semantic reasoning, budgets, provenance).
- Section 4 (Methods) , the typed PlanGraph (4.1), token optimization (4.2), tool optimization (4.3), the prism / ellipse / MOBR / DRM math (4.4), STAGD-DRM and TGARD / DC-TGARD (4.5), the S-KBM validator (4.6), MCP and A2A (4.7).
- Section 5 (Experiments) , the golden dataset and the latency / token / cost table (reproduced live; see `evaluation/eval_results/` and `evaluation/ablation_results/`).
- Section 6 (Discussion) , the honest limitations.

---

## 7. What evaluation still needs doing (the next phase)

Stated plainly so the work is clear:

1. **Reproduce the numbers (done).** `offline_eval` was run end to end with a live key on Claude Sonnet 4.6; the `evaluation/eval_results/<timestamp>.{md,json}` report, the cache ablation (`evaluation/ablation_results/`), and the validator audit (`evaluation/validator_audit_results/`) are checked in. Remaining: grow the golden set beyond 3 queries and wire the in-flight tool deduplicator into the run path.
2. **Judge the free-text summary.** Add a calibrated LLM-as-judge (with a small human-labeled set) for the one part the deterministic oracle does not cover.
3. **Train the anomaly head.** Load or train a Pi-DPM checkpoint on the live trajectory distribution so the Abnormal Gap Measure's data term is real, not default-initialized.
4. **Grow and stratify the golden set.** Three anchor cases prove the harness; a larger, stratified set (by domain, difficulty, region count) makes the pass-rate meaningful.

**Honesty rails (stated in the paper and on the site):** GeoTrace-Agent is a research-engineering blueprint with production-shaped architecture. The efficiency numbers are reproduced from a live golden-set run and a cache ablation (checked in under `evaluation/`), on a small 3-query golden set; the in-flight tool deduplicator is implemented but not yet wired into the run path. It is not running at production traffic.

---

## 8. The Ankur angle (agent evaluation infrastructure)

For a Google Cloud AI evaluation-infrastructure lead, lead with sections 3.8, 3.9, and 2 (the eval surfaces, the deterministic-oracle choice, and the analogy table). The single most credible thing to point at is the **public `evaluation/` and `observability/` code**, the golden-set CI gate, the online drift monitor, the OpenTelemetry tracer with tail sampling, and the per-stage cost ledger. The honest, senior move is to name the open piece up front (no LLM judge on the free-text summary yet) and frame it as a peer conversation about how a platform like ADK scores trajectories and calibrates judges.
