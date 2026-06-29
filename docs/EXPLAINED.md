# GeoTrace-Agent, Explained

> A plain-language tour of what the system actually does and, more importantly, **why each design decision was made**. Companion to the [project website](https://arunshar.com) and the [full paper](https://arunshar.com/projects/geotrace-agent/).

---

## 0. The one mental model

GeoTrace-Agent answers natural-language questions about moving objects ("could these two vessels have met near here, in this window?"). The single idea that organizes everything:

> **The LLM plans. Deterministic kernels compute. A validator gates. The LLM never decides anything that has a ground truth.**

Think of it as **a compiler and a type system wrapped around a language model.** The LLM writes a program (a plan); the runtime type-checks that program, runs the parts that are exact math with exact math, and refuses to return any answer that is physically impossible. The LLM is used only where judgment is genuinely needed (deciding the steps, narrating the result), not where the answer is decidable by geometry.

Everything below is downstream of that one decision.

---

## 1. How the agents actually interact

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

It encodes several decisions at once:

- **Topo layers + `asyncio.gather`**: nodes with no remaining dependencies run *concurrently*. Computing two prisms is two parallel tasks, not two sequential calls.
- **`return_exceptions=True`**: one node failing does not crash the run; the orchestrator records the error and continues, except for `KinematicViolation`, which is re-raised immediately (an impossible answer must abort, not degrade).
- **Budget enforced after every layer**: token, tool-call, and wallclock budgets are checked between layers; an overrun sets `terminated_by_budget` and raises `BudgetExceeded`. The agent can never run away.
- **Per-stage ledger**: each stage runs inside `_StageContextManager`, which opens an OpenTelemetry span and appends a `StageTrace` (tokens in/out, cost, cache-hit). That ledger is how cost is attributed per query.

`execute_node` is a dispatch table on the node's typed `kind`: `PRISM` to the SpaceTimeReasoner, `GAPS` to the GapDetector, `TGARD/DC_TGARD` to the RendezvousFinder, `VALIDATE` to the ValidatorAgent, `SUMMARIZE` handled at the end. Each agent has a typed input and a typed output, so the orchestrator never parses free text to decide what to do next.

---

## 2. The design decisions and the rationale (the heart of it)

Each decision below is stated as **what, why, and what the alternative would have cost.**

### 2.1 A typed PlanGraph instead of free-form chain-of-thought
**What.** The planner (`app/agents/planner.py`) does not emit prose. It returns a JSON object validated against a strict schema: a DAG of nodes, each one of `prism.compute`, `gaps.detect`, `rendezvous.tgard`, `rendezvous.dc_tgard`, `validate.kinematic`, `retrieve.semantic`, `summarize`, with `deps`, `inputs`, `expected_tokens`, and a `confidence_prior`.
**Why.** Four properties fall out for free: (a) **auditable**, every reasoning step is a typed record you can inspect; (b) **replayable**, the plan plus the prompt version reproduce the run exactly; (c) **parallelizable**, a DAG has topo layers, so independent steps run concurrently; (d) **bounded**, `sum(expected_tokens)` is checked against the budget *before* any work runs (`PlanInfeasible` otherwise). It also makes caching effective, because a typed plan has stable keys.
**Alternative cost.** Free-form CoT is a string you cannot statically check, cannot safely parallelize, and cannot bound. You would be parsing prose to decide control flow, the thing that makes agents flaky.

### 2.2 Deterministic kernel before the LLM
**What.** Every spatially-decidable sub-problem is computed by a numerical kernel (`app/components/space_time_prism.py`), not the LLM. The space-time prism, the geo-ellipse, the bounding rectangle, the prism intersection, all exact.
**Why.** LLMs hallucinate distances and skip the physics check. Reachability has a *ground truth*: given two anchors and a max speed, the set of reachable points is a geo-ellipse with semi-major axis `a = 0.5 * v_max * duration`. There is no reason to ask a model to approximate something you can compute exactly, reproducibly, and cheaply.
**Alternative cost.** Trusting the LLM for geometry means occasional confident-but-wrong answers, which in maritime safety or homeland security is unacceptable.

### 2.3 A single `TokenOptimizer` choke-point
**What.** Every LLM call in the system goes through one class (`app/services/token_optimizer.py`). It owns five jobs: prompt compression, prefix-cache-aware assembly (`cache_control: ephemeral`), structured-output enforcement with one delta-correction retry, a per-call `max_tokens` clamp against the remaining budget, and cost accounting (returns `tokens_in, tokens_out, cost_usd, cache_hit`).
**Why.** Cross-cutting concerns belong in one place. Budget, caching, retries, timeouts, and cost attribution are *uniform* because there is exactly one door every call walks through. The orchestrator can never overshoot the budget on a planner regression, because the clamp lives at the door.
**Alternative cost.** Scattering LLM calls means scattering budget logic, cost accounting, and retry policy, which drift apart and leak spend.

### 2.4 The efficiency stack
**What.** (1) **Adaptive prompt compression**, prompts over a threshold are head-tail truncated with an elision marker. (2) **Hybrid semantic cache**, an exact-key lookup layered over a near-key embedding lookup, so identical anchor pairs return in O(1) and near-duplicate questions reuse prior answers. (3) **In-flight tool deduplication**, the `ToolBatcher` keys concurrent calls by `(tool, sha256(args))` and collapses duplicates into one awaitable; it is implemented but not yet wired into the orchestrator run path, so it does not contribute to the measured numbers.
**Why.** The two active layers are complementary: the cache eliminates LLM cost on repeated or near-duplicate queries, and compression shrinks each call. On a live golden-set run with Claude Sonnet 4.6 the full pipeline costs about 0.026 USD per query (~2,736 tokens), and a repeated query drops to zero LLM cost on a cache hit (cold to warm; see `evaluation/eval_results/` and `evaluation/ablation_results/`). The cache does not fire on novel queries, so there is no blanket per-query reduction.
**Alternative cost.** Any one layer alone leaves money on the table; the dedup-vs-cache distinction (in-run duplicates vs cross-run repeats) is the subtle part most systems miss.

### 2.5 The kinematic validator as a hard gate (and a subtle rationale)
**What.** `app/agents/validator.py` is the last thing every region passes through. It enforces the Hagerstrand feasibility condition on the **observations**: consecutive anchors must be mutually reachable, `dist(A, B) <= v_max * (t_B - t_A)` (with a 5% tolerance). Failure raises `KinematicViolation`, surfaced as HTTP 422.
**Why this exact form.** The code comment captures a real lesson: an earlier version gated on the rendezvous region's **bounding-box diagonal divided by the meet window** as a "required speed." That was wrong, a region is the *set of alternative meeting points*, not a path to traverse, and its extent scales with the prism duration while the meet window is shorter, so that proxy rejected every valid region. The fix gates the physics where the physics actually lives (the observations), not on a derived artifact.
**Why a hard gate, not a soft penalty.** Physical feasibility is an invariant, not a preference. A soft score could be overridden by a confident summary; a hard gate cannot. The system would rather return nothing than return an impossible region.
**Alternative cost.** A soft penalty, or no gate, lets physically impossible answers reach users, the failure mode that erodes trust fastest.

### 2.6 MCP for tools, A2A for agents
**What.** The prism kernel is exposed as a Model Context Protocol server (`prism.compute`, `prism.intersect`, `prism.merge_dynamic`); the orchestrator advertises a capability card and accepts JSON-RPC 2.0 Agent-to-Agent calls.
**Why.** The geometry kernel is genuinely reusable; making it an MCP server means any MCP-aware client, IDE plugin, or sibling agent can call it without going through this app. A2A makes GeoTrace-Agent a first-class citizen other agents can invoke. This is the same reason you put a typed RPC interface on a useful service.
**Alternative cost.** Burying the kernel behind one app's HTTP surface makes it un-reusable and un-composable with the broader agent ecosystem.

### 2.7 OpenTelemetry plus a per-stage cost ledger
**What.** `observability/tracer.py` wraps every tool call in a span carrying `tool.name, cache_hit, cost_usd, tokens_in/out`; traces are **tail-sampled** (100% of errors, 1% of successes). `observability/cost_tracker.py` writes one row per `(trace_id, stage)`.
**Why.** You cannot optimize what you cannot measure. Per-stage attribution is what made the efficiency work legible (the reproduced per-query cost and the cache cold-to-warm saving came from this ledger, not a guess). Tail sampling keeps every failure for debugging while bounding storage, the right default for an eval stack.
**Alternative cost.** Aggregate metrics tell you the system is slow or expensive but not *where*; you cannot do targeted optimization or honest cost claims without per-stage traces.

### 2.8 Offline golden-set CI gate plus online drift monitor
**What.** Offline (`evaluation/offline_eval.py`): replay a labeled golden set, score structure, tool-trajectory (which algorithm), feasibility, tightness, tokens, latency, and **fail CI if pass-rate < 0.8**. Online (`evaluation/online_monitor.py`): sample 1% of traffic, recompute the validator independently, re-run a frozen-prompt plan and alert on **plan drift**.
**Why.** Evaluation is a discipline, not an afterthought. The CI gate means a prompt or model change that regresses the agent fails the build. The drift monitor catches the silent case, a new model quietly changes the plan distribution on live traffic.
**Alternative cost.** Without a gate, regressions ship; without a drift monitor, you find out from users.

### 2.9 A deterministic feasibility oracle instead of an LLM judge
**What.** The core evaluator is the prism feasibility oracle, not an LLM-as-judge. A claimed region is valid only if it sits inside the physically reachable prism.
**Why.** The domain has ground truth, so a deterministic check is reproducible, has zero judge variance, and is cheap enough to run on 100% of outputs as a guardrail. An LLM judge is itself a model that drifts across versions and needs its own calibration.
**Honest limit.** This covers the structured, checkable parts. It does **not** judge the free-text summary's quality or faithfulness; that is exactly where a calibrated LLM judge belongs, and it is the open piece.

### 2.10 HITL queue that closes the loop to RL
**What.** Traces below a confidence threshold flow to a Postgres human-in-the-loop queue; reviewer verdicts export as preference triples for the sibling **Pi-GRPO** project (DPO / GRPO post-training).
**Why.** It connects agentic reasoning to reward-modeled fine-tuning without changing the LLM call surface or the validator. Human judgment becomes training signal.
**Alternative cost.** A dead-end review queue collects labels nobody uses; wiring it to Pi-GRPO turns review into model improvement.

---

## 3. Core-file walkthrough (the ~8 files that matter)

- **`app/services/orchestrator.py`** , the conductor. `run()` is the request lifecycle; `_execute_plan` is the topo-layer parallel loop; `_enforce_budget` is the governor; `_StageContextManager` is the span + ledger. Start here.
- **`app/agents/planner.py`** , the only place the LLM decides control flow. Emits the typed PlanGraph against `_PLAN_JSON_SCHEMA`; `_coerce_plan` validates the DAG, attaches anchors to `prism.compute` nodes, runs the cycle check, and enforces the token budget.
- **`app/models.py`** , the type system: `PlanGraph` / `PlanNode` / `PlanNodeKind`, `topo_layers()`, `Budget`, `RendezvousRegion`, `GeoEllipse`. These types are the contract every agent honors.
- **`app/components/space_time_prism.py`** , the geometry IP. `Prism.compute` (semi-major `0.5 * v_max * duration`, feasibility `d_ab <= L`), `ellipse_at(t)`, `mobr`, `ellipse_polygon`, `merge_dynamic` (DRM), and `intersect` (time-slice prism intersection, the operation TGARD rides on). Note `to_payload`/`from_payload`, lossless serialization for durable-execution activity boundaries.
- **`app/services/token_optimizer.py`** , the single LLM door (section 2.3). Also holds the offline stub that returns a deterministic plan when no API key is set, which is why the demo runs without a key.
- **`app/agents/validator.py`** , the hard kinematic gate (section 2.5); short, but the rationale comment is the most instructive in the codebase.
- **`evaluation/offline_eval.py` + `evaluation/online_monitor.py`** , the two eval surfaces (section 2.8).
- **`observability/tracer.py` + `observability/cost_tracker.py`** , spans, tail sampling, and the per-stage cost ledger (section 2.7).
- **`spaces/hf-demo/streamlit_app.py`** , the demo (section 4).

The other agents (`space_time_reasoner.py`, `gap_detector.py`, `rendezvous_finder.py`) are thin, typed clients over the prism kernel; the gap detector additionally fuses a Pi-DPM anomaly term into the Abnormal Gap Measure when torch is available, and falls back to a deterministic surrogate otherwise.

---

## 4. The demo, and how to read it

The Hugging Face Space (`spaces/hf-demo/streamlit_app.py`) is the system in miniature, CPU-only, no database. **Sidebar:** pick a preset (rendezvous / gap audit / prism-only / coverage-gap on a track), edit the question, the vessel anchors (lat, lon, time), and a budget; click Run. **Main panel, four views of the same run:**

1. **Chain of thought (typed PlanGraph)** , the plan rendered as a flowchart. This is the agent's "program."
2. **Map** , the space-time prism ellipses, their bounding rectangles, and (in green) the candidate rendezvous region. This is the geometry the kernel computed.
3. **Validator** , regions returned, methods used, and `kinematic_pass`. This is the gate's verdict.
4. **Cost / tokens** and the raw **Result JSON** , what the run spent and produced.

The panel shows the *machinery* (plan + map + JSON) rather than a single narrated paragraph. Without an API key it runs the deterministic stub, so the geometry always renders even with no LLM. A one-line "what just happened" summary panel would make it read more easily.

---

## 5. What the demo shows, preset by preset (a live read)

Running the presets against the live planner (Claude Sonnet 4.6) is the fastest way to see the design decisions act. The thing to watch is that **the same system emits a different typed plan for each question**, which is the whole point of 2.1.

- **Prism-only** (a flat ask). The minimum plan, `prism.compute -> summarize`, two nodes. One prism, no gap detector, no validator step. Cheapest run, about $0.008 (~745 tokens in, ~379 out). It proves the planner does not pad: a simple question gets a simple program.
- **Trajectory-gap audit** (two anchors only). A four-node plan, `prism.compute -> gaps.detect -> validate.kinematic -> summarize`. One prism, and **zero gaps**, because the input is a single anchor pair, not a track; gap detection looks for an abnormal jump *between consecutive samples*, and two points have no interior to inspect. The agent reasoned correctly; the input simply contains no gap. About $0.016.
- **Coverage-gap on a vessel track** (the input that makes it light up). The same four-node shape, but fed an eight-ping AIS track with a deliberate ~2-hour blackout in the middle. The detector finds **one abnormal gap** (118 minutes, 33 km) and scores it with the Abnormal Gap Measure, `AGM = lambda*(1 - P_phys) + (1 - lambda)*P_data` with `lambda = 0.6`; here P_phys = 1.00 (the jump is kinematically plausible at vessel speed) and P_data = 0.86 (off the normal coverage cadence), for an **AGM of 0.345**, and the map paints the coverage polygon over the blackout in amber. This is the preset that actually exercises STAGD-DRM and the AGM, and it makes the row above concrete: *the detector is only as informative as the trajectory you feed it.*
- **Rendezvous between two vessels** (two anchor pairs). The largest plan, five nodes: two `prism.compute`, then `rendezvous.dc_tgard` to intersect the two reachable envelopes, then `validate.kinematic`, then `summarize`. Result: **one rendezvous region**, method DC-TGARD, `kinematic_pass: true`, confidence about 0.28. The green polygon is the set of points where the two vessels *could* have met; the modest confidence honestly reflects a small envelope overlap. Most expensive, about $0.020.

**The cross-cutting reads.** (1) Four questions, four distinct PlanGraph shapes (2, 4, 4, 5 nodes): the typed planner routes by intent, it is not a fixed template. (2) Cost is attributed per query from the stage ledger and scales with plan complexity ($0.008 < $0.016 < $0.020); a repeated query drops to zero LLM cost on a cache hit. (3) The validator gates: a region is returned only when one is both found and physically feasible, and zero otherwise, never a plausible-but-impossible answer.

**Worst cases and honest caveats.**
- **Empty by construction, not by failure.** The two-anchor gap audit returns no gaps, and the prism-only and gap presets return no regions. That is correct: there is nothing to find in those inputs. The populated panels are the rendezvous region and the track-gap polygon, where the input actually contains the thing being detected.
- **Passing the gate is not high confidence.** The rendezvous region clears the hard kinematic gate yet scores ~0.28. Passing means *physically possible*, not *likely*; confidence is a separate, softer signal, and conflating the two would overclaim.
- **The anomaly term is a placeholder.** The AGM's P_data (the Pi-DPM term) runs on a default-initialized head unless a trained checkpoint is loaded, so today the score leans on the kinematic term P_phys. Training that head on the live trajectory distribution is item 3 of section 7.
- **Illustrative, not powered.** These are single runs, not a benchmark; the reproduced cost numbers come from a 3-query golden set. The honesty rails in section 7 apply.

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
