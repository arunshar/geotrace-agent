# Agents: Capability Cards

Each agent in GeoTrace-Agent is a first-class object. It has a capability card, an A2A endpoint, and a contract that the orchestrator can rely on. Cards live in `app/a2a/cards/` as YAML and are mirrored here for human review.

## PlannerAgent (`app/agents/planner.py`)

| Field | Value |
|---|---|
| capability | `plan.decompose` |
| inputs | `Question`, `Budget`, `ConversationState` |
| outputs | `PlanGraph` (DAG of `PlanNode`) |
| model | `claude-sonnet-4-6` (default) |
| reasoning style | Chain-of-thought with explicit physical-reasoning checkpoints |
| failure modes | infeasible plan, exceeding `max_tools`, ambiguous question (escalates to HITL) |

Notes. The planner emits a typed plan, not free-form text. Each `PlanNode` declares the agent it will call, the tool budget it expects, and a confidence prior. The orchestrator topo-sorts the DAG and runs nodes with no remaining dependencies in parallel.

## SpaceTimeReasoner (`app/agents/space_time_reasoner.py`)

| Field | Value |
|---|---|
| capability | `prism.compute`, `rendezvous.candidates` |
| inputs | `AnchorPair`, `SpeedBounds`, `TimeSlices` |
| outputs | `Prism`, `GeoEllipse[]`, `MOBR[]` |
| model | none (deterministic numerical kernel) |
| failure modes | numerical instability for tiny ellipses (clamped to floating-point epsilon), unreachable anchors (returns empty prism with a `reason`) |

This is the deterministic geometric kernel. It does not call an LLM.

## GapDetectorAgent (`app/agents/gap_detector.py`)

| Field | Value |
|---|---|
| capability | `gaps.detect`, `gaps.merge` (DRM) |
| inputs | `Trajectory`, `coverage_threshold`, `merge_radius_km` |
| outputs | `Gap[]` with `AbnormalGapMeasure` |
| model | optional `Pi-DPM` reconstruction-error scorer |
| extends | STAGD + Dynamic Region Merge from Sharma et al., ACM TIST 2024. |

## RendezvousFinderAgent (`app/agents/rendezvous_finder.py`)

| Field | Value |
|---|---|
| capability | `rendezvous.tgard`, `rendezvous.dc_tgard` |
| inputs | `Gap[]`, `RoadNetwork`, `Anchors` |
| outputs | `RendezvousRegion[]` with tightened bounds |
| extends | TGARD + DC-TGARD from Sharma et al., SIGSPATIAL 2022. |

## ValidatorAgent (`app/agents/validator.py`)

| Field | Value |
|---|---|
| capability | `validate.kinematic`, `validate.consistency` |
| inputs | candidate `RendezvousRegion[]`, `KinematicBounds` |
| outputs | `Validation` (pass / fail with reason) |
| guarantees | every region returned to the user has been validated |

## Tools (registered in `app/agents/tools/__init__.py`)

| Tool | MCP server | Capability |
|---|---|---|
| `prism.compute` | `mcp_servers/prism_mcp.py` | Hägerstrand prism + geo-ellipse + MOBR |
| `ais.history` | `mcp_servers/ais_mcp.py` | trajectory snippets from MarineCadastre / ESRI maritime feed |
| `road.network` | `mcp_servers/road_network_mcp.py` | OSM extract + travel-time bounds |
| `weather.fetch` | `mcp_servers/weather_mcp.py` | Copernicus CDS sea state + wind |
| `vector.search` | `agents/tools/vector_search.py` | Chroma over historical incident corpus |
| `web.search` | `agents/tools/web_search.py` | OSINT for vessel registries |
| `code.search` | `agents/tools/code_search.py` | grep over our own playbook repository |

## A2A protocol

Inter-agent calls go over JSON-RPC 2.0 (`app/a2a/protocol.py`). Each agent advertises its capabilities at `GET /a2a/.well-known/capabilities` (an Agent Card). The orchestrator caches capability cards for 60 s. Calls carry a `trace_id` that propagates to OTEL.

## HITL surface

Items with `confidence < 0.7` or `validator.failed = true` are pushed to a Postgres queue read by the Streamlit reviewer console. A reviewer's verdict is written back as `feedback.label` and is consumed by the offline evaluator and (optionally) by a future RL fine-tuner (see sibling project `pi-grpo`).
