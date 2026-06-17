# GeoTrace on Temporal (and NeMo): durable, replayable spatial agents

This package ports `app/services/orchestrator.Orchestrator.run` onto Temporal. The
point is not "add a queue"; it is that GeoTrace's neuro-symbolic boundary IS a
determinism boundary, and Temporal's workflow/activity split is the same boundary,
so the port is almost a one-to-one mapping rather than a rewrite.

## The one idea

A GeoTrace run interleaves two kinds of step:

- Nondeterministic, model-driven: the LLM planner that decomposes a question into
  a plan graph, the gap detector, the answer synthesizer. Same input, possibly
  different output. These are the "neuro" steps.
- Deterministic, symbolic: the space-time prism, TGARD / DC-TGARD rendezvous
  detection, kinematic validation, the budget guard, confidence aggregation. Same
  input, same output, every time. These are the "symbolic" steps.

Temporal demands exactly this split. Workflow code must be deterministic so it can
be rebuilt by replaying its event history; anything nondeterministic or
side-effecting must be an activity, whose result is recorded once and replayed
verbatim. So:

| GeoTrace step | Nature | Temporal placement | Why |
|---|---|---|---|
| Plan-graph topo schedule, budget guard, confidence aggregation, HITL gate | deterministic symbolic | workflow (`workflows.py`) | must replay identically; no I/O |
| LLM planner, gap detector, summarizer | nondeterministic | activity | recorded once so replay never re-calls the model and never drifts |
| Space-time prism, TGARD/DC-TGARD, kinematic validation | deterministic symbolic | activity (by necessity, not by nature) | pure, but imports heavy geo/numeric code that must stay out of the workflow sandbox |
| Semantic-cache read, HITL enqueue | side effect | activity, at-least-once, idempotent | external state; dedup on the workflow id |

The symbolic geo kernels are deterministic, so in principle they could run inline
in the workflow; they are activities only to keep the deterministic sandbox free
of heavy imports. That distinction (deterministic-but-heavy vs truly
nondeterministic) is the senior point: the LLM steps MUST be activities for
correctness, the geo steps are activities for hygiene.

## What the port buys you

- Crash safety for free. A worker dying mid-run loses nothing: the plan, every
  completed node, and the synthesized answer are in history, and a redelivered
  activity that already finished is replayed as its recorded result, not re-run.
- Durable human-in-the-loop. The orchestrator's low-confidence path is a
  fire-and-forget `hitl.enqueue`. Here the workflow enqueues and then
  `wait_condition`s on a human-approval signal (`review`), durably parking the run
  for as long as a human takes (minutes or days) and surviving restarts. That
  long-lived, human-gated, exactly-once-side-effect run is precisely what
  Temporal's AI SDK productizes.
- Effectively-once side effects. Activities are at-least-once, so each is
  idempotent (the HITL enqueue dedups on the durable workflow id). Effectively-once
  equals at-least-once delivery plus idempotency; true exactly-once execution of
  arbitrary side effects is not deliverable, and the code is honest about that.
- A live, queryable run. `progress` is a synchronous query of stage/tokens/cost
  with no mutation; `review` is the async signal in.

## Determinism discipline (what the workflow must not do)

- No wall clock or randomness: the workflow uses `workflow.now()` for the
  wallclock budget and `workflow.uuid4()` for the trace id.
- No I/O and no model calls in workflow code: all of it is in activities.
- The plan graph's `topo_layers()` is a pure function, so it is safe to run inline
  in the workflow to schedule the fan-out.
- Activity timeouts and a bounded retry policy are set per call; the LLM steps get
  a longer deadline than the geo kernels.

## Files

- `models.py` serializable contracts that cross the activity boundary (so the
  workflow only ever sees flat, version-stable data, never a live prism handle).
- `activities.py` the nondeterministic and side-effecting steps, wrapping the real
  `Orchestrator` components so the durable port reuses the existing agents.
- `workflows.py` `GeoTraceWorkflow`: the deterministic control plane plus the
  `review` signal and `progress` query.
- `worker.py` / `run_client.py` host and drive it.
- `tests/test_temporal_workflow.py` runs the workflow end to end against mocked
  activities in Temporal's time-skipping test environment, covering the
  high-confidence path and the durable HITL-signal path.

## Mapping to NVIDIA NeMo (the genAI agent stack, not PhysicsNeMo)

Temporal owns durability and control; NeMo owns the model and agent surface. They
compose:

- NeMo Agent Toolkit: the planner and synthesizer activities call NeMo-served
  models; the agent/tool definitions live in NeMo, their durable orchestration in
  the Temporal workflow.
- NeMo Guardrails: input/output rails back the `guard` and `output_filter`
  activities (the existing `app/security` filters are the seam).
- NeMo Retriever: backs the `RETRIEVE` node and the existing `hybrid_retriever`.
- NIM serving on MSI: the planner/gap/summarize activities target a NIM endpoint;
  because they are activities, a slow or failed NIM call is retried and recorded
  without breaking workflow determinism.

The boundary stays clean: NeMo produces tokens (nondeterministic, so always behind
an activity), Temporal decides what runs, in what order, how it retries, and when
it waits for a human.

## Honest scope

Temporal is not running in production for GeoTrace today; this is the designed and
tested port, defensible on first principles. The workflow logic, the determinism
boundary, and the durable HITL signal are real and covered by the test against
mocked activities. The remaining production wiring is the per-node-kind
serialization of intermediate symbolic objects (for example a prism handed from a
PRISM node to a TGARD node), which `models.NodeResult.payload` is the contract for,
and pointing the activities at live NeMo/NIM endpoints and the real cache/HITL
stores.
