"""Standalone Streamlit demo of GeoTrace-Agent for the Hugging Face Space.

Free-tier (CPU-only, 16 GB RAM) deployment. Imports the project's core
Python modules directly (no FastAPI / Postgres / Redis / Chroma) and
keeps state in memory.

If `ANTHROPIC_API_KEY` is set as a Space secret, the live planner
runs; otherwise the offline stub returns a deterministic plan.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Any

import streamlit as st
from shapely.geometry import mapping

# The `app/` package is vendored alongside this file (see spaces/hf-demo/app/),
# so it imports cleanly without sys.path manipulation.
from app.agents.gap_detector import GapDetectorAgent
from app.agents.planner import PlannerAgent
from app.agents.rendezvous_finder import RendezvousFinderAgent
from app.agents.space_time_reasoner import SpaceTimeReasoner
from app.agents.validator import ValidatorAgent
from app.components.space_time_prism import Prism
from app.config import get_settings
from app.models import (
    Anchor,
    AnchorPair,
    Budget,
    ConversationState,
    PlanNodeKind,
    QueryIn,
    RendezvousRegion,
)
from app.services.semantic_cache import SemanticCache
from app.services.token_optimizer import TokenOptimizer

st.set_page_config(page_title="GeoTrace-Agent", page_icon="🛰️", layout="wide")


@st.cache_resource(show_spinner=False)
def _event_loop() -> asyncio.AbstractEventLoop:
    """One persistent event loop, reused across Streamlit reruns.

    Streamlit reruns the whole script on every interaction. Calling
    ``asyncio.run`` each time creates and then closes a fresh loop, but the
    cached httpx.AsyncClient lives on across reruns, so on the second run it
    tries to close connections bound to the first (now closed) loop and raises
    ``RuntimeError: Event loop is closed``. Keeping a single loop alive and
    running everything on it removes that mismatch.
    """
    return asyncio.new_event_loop()


@st.cache_resource(show_spinner=False)
def _bootstrap() -> dict[str, Any]:
    """Build a single in-process copy of the agent stack."""

    # propagate the HF Space secret into the project's settings
    if "ANTHROPIC_API_KEY" in os.environ and not os.environ.get("GT_ANTHROPIC_API_KEY"):
        os.environ["GT_ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]

    settings = get_settings()
    cache = _event_loop().run_until_complete(SemanticCache.connect(settings))
    token_opt = TokenOptimizer(settings, cache=cache)
    planner = PlannerAgent(settings, token_opt)
    st_reasoner = SpaceTimeReasoner(settings)
    gap_det = GapDetectorAgent(settings, token_opt)
    rdv = RendezvousFinderAgent(settings, st_reasoner)
    val = ValidatorAgent(settings)
    return {
        "settings": settings, "cache": cache, "token_opt": token_opt,
        "planner": planner, "st": st_reasoner, "gap": gap_det,
        "rdv": rdv, "val": val,
    }


# ---------------------------------------------------------------------------
# Sidebar: question, anchors, budget
# ---------------------------------------------------------------------------

PRESETS = {
    "Rendezvous between two vessels": {
        "question": ("Could VESSEL-1234 have rendezvoused with VESSEL-9876 "
                     "between 06:00Z and 12:00Z near 56N 162W?"),
        "anchors": [
            (56.10, -162.05, "2026-01-15T06:00:00Z"),
            (56.30, -162.40, "2026-01-15T12:00:00Z"),
            (56.12, -162.08, "2026-01-15T06:00:00Z"),
            (56.28, -162.34, "2026-01-15T12:00:00Z"),
        ],
        "domain": "vessel",
    },
    "Trajectory-gap audit": {
        "question": ("Did VESSEL-1234 have a coverage gap consistent "
                     "with signal denial near the Aleutian shelf on 2026-01-15?"),
        "anchors": [
            (56.10, -162.05, "2026-01-15T06:00:00Z"),
            (56.30, -162.40, "2026-01-15T12:00:00Z"),
        ],
        "domain": "vessel",
    },
    "Compute prism only": {
        "question": ("Compute the prism between (56.10, -162.05, 06:00Z) "
                     "and (56.30, -162.40, 12:00Z)."),
        "anchors": [
            (56.10, -162.05, "2026-01-15T06:00:00Z"),
            (56.30, -162.40, "2026-01-15T12:00:00Z"),
        ],
        "domain": "vessel",
    },
    "Coverage-gap on a vessel track": {
        "question": ("VESSEL-1234 reported AIS along the Aleutian shelf on 2026-01-15, "
                     "then went dark for about two hours. Audit its track for a coverage "
                     "gap consistent with signal denial and score how abnormal it is."),
        "anchors": [
            (56.05, -162.10, "2026-01-15T06:00:00Z"),
            (56.34, -161.42, "2026-01-15T08:46:00Z"),
        ],
        "domain": "vessel",
        # An 8-ping AIS track: ~8 min between pings, then a ~2-hour blackout
        # (06:32 -> 08:30) the gap detector flags as one abnormal gap.
        "track": [
            (56.05, -162.10, "2026-01-15T06:00:00Z"),
            (56.07, -162.06, "2026-01-15T06:08:00Z"),
            (56.09, -162.02, "2026-01-15T06:16:00Z"),
            (56.11, -161.98, "2026-01-15T06:24:00Z"),
            (56.13, -161.94, "2026-01-15T06:32:00Z"),
            (56.30, -161.50, "2026-01-15T08:30:00Z"),
            (56.32, -161.46, "2026-01-15T08:38:00Z"),
            (56.34, -161.42, "2026-01-15T08:46:00Z"),
        ],
    },
}


with st.sidebar:
    st.title("🛰️ GeoTrace-Agent")
    st.caption(
        "Production-grade agentic AI for spatiotemporal trajectory reasoning. "
        "Hägerstrand space-time prisms x multi-agent chain of thought x MCP / A2A."
    )

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GT_ANTHROPIC_API_KEY"))
    if has_anthropic:
        st.success("Live Anthropic planner enabled.")
    else:
        st.info("No `ANTHROPIC_API_KEY` set; running with the deterministic offline stub.")

    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    preset = PRESETS[preset_name]

    question = st.text_area("Question", preset["question"], height=80)
    domain = st.selectbox("Domain", ["vessel", "vehicle", "pedestrian", "uav"],
                          index=["vessel", "vehicle", "pedestrian", "uav"].index(preset["domain"]))

    st.markdown("**Vessel 1: Anchor A**")
    a_lat = st.number_input("lat", value=preset["anchors"][0][0], format="%.4f", key="a_lat")
    a_lon = st.number_input("lon", value=preset["anchors"][0][1], format="%.4f", key="a_lon")
    a_t = st.text_input("time (ISO8601, UTC)", value=preset["anchors"][0][2], key="a_t")

    st.markdown("**Vessel 1: Anchor B**")
    b_lat = st.number_input("lat", value=preset["anchors"][1][0], format="%.4f", key="b_lat")
    b_lon = st.number_input("lon", value=preset["anchors"][1][1], format="%.4f", key="b_lon")
    b_t = st.text_input("time (ISO8601, UTC)", value=preset["anchors"][1][2], key="b_t")

    use_second_pair = len(preset["anchors"]) >= 4
    if use_second_pair:
        st.markdown("**Vessel 2: Anchor A**")
        c_lat = st.number_input("lat", value=preset["anchors"][2][0], format="%.4f", key="c_lat")
        c_lon = st.number_input("lon", value=preset["anchors"][2][1], format="%.4f", key="c_lon")
        c_t = st.text_input("time (ISO8601, UTC)", value=preset["anchors"][2][2], key="c_t")

        st.markdown("**Vessel 2: Anchor B**")
        d_lat = st.number_input("lat", value=preset["anchors"][3][0], format="%.4f", key="d_lat")
        d_lon = st.number_input("lon", value=preset["anchors"][3][1], format="%.4f", key="d_lon")
        d_t = st.text_input("time (ISO8601, UTC)", value=preset["anchors"][3][2], key="d_t")

    st.markdown("**Budget**")
    max_tokens = st.slider("max tokens", 1_000, 32_000, 8_000, step=1_000)
    max_tools = st.slider("max tools", 1, 16, 6)
    max_seconds = st.slider("max seconds", 5, 60, 25)

    run = st.button("Run", type="primary", width="stretch")


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------


def _parse_t(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)


def _plan_to_mermaid(plan: Any) -> str:
    """Render a PlanGraph as a Mermaid flowchart."""

    lines = ["flowchart TD"]
    for n in plan.nodes:
        label = f"{n.id}<br/>{n.kind.value}<br/>conf={n.confidence_prior:.2f}"
        lines.append(f'    {n.id}["{label}"]')
    for n in plan.nodes:
        for d in n.deps:
            lines.append(f"    {d} --> {n.id}")
    return "\n".join(lines)


def _ellipse_to_geojson(prism: Prism) -> dict:
    return mapping(prism.ellipse_polygon())


def _mobr_to_geojson(prism: Prism) -> dict:
    return mapping(prism.mobr())


async def _run(state: dict[str, Any], q: QueryIn,
               track: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    plan = await state["planner"].plan(q, ConversationState())
    results: dict[str, Any] = {}
    prisms: list[Prism] = []
    gaps: list[Any] = []

    for layer in plan.topo_layers():
        for node in layer:
            if node.kind is PlanNodeKind.PRISM:
                prism_res = await state["st"].compute(AnchorPair(**node.inputs["pair"]), q.domain)
                prisms.append(prism_res.prism)
                results[node.id] = prism_res
            elif node.kind is PlanNodeKind.GAPS:
                gap_inputs = dict(node.inputs)
                if track:  # demo presets can supply a real multi-point track to audit
                    gap_inputs["trajectory"] = track
                    gap_inputs.setdefault("domain", q.domain)
                gap_out = await state["gap"].detect(gap_inputs)
                gaps.extend(gap_out)
                results[node.id] = gap_out
            elif node.kind in (PlanNodeKind.TGARD, PlanNodeKind.DC_TGARD):
                upstream = [results[d].prism for d in node.deps if hasattr(results.get(d), "prism")]
                method = "TGARD" if node.kind is PlanNodeKind.TGARD else "DC-TGARD"
                results[node.id] = await state["rdv"].find(upstream, method=method)
            elif node.kind is PlanNodeKind.VALIDATE:
                candidates = [
                    r
                    for d in node.deps
                    for r in (results.get(d) if isinstance(results.get(d), (list, tuple)) else [])
                    if isinstance(r, RendezvousRegion)
                ]
                results[node.id] = await state["val"].validate(candidates, domain=q.domain)
            elif node.kind is PlanNodeKind.SUMMARIZE:
                results[node.id] = None

    validated_region_values = [
        results[node.id]
        for node in plan.nodes
        if node.kind is PlanNodeKind.VALIDATE and isinstance(results.get(node.id), list)
    ]
    region_values = validated_region_values or [v for v in results.values() if isinstance(v, list)]
    regions = [
        r
        for value in region_values
        for r in (value or [])
        if isinstance(r, RendezvousRegion)
    ]
    return {
        "plan": plan,
        "prisms": prisms,
        "gaps": gaps,
        "regions": regions,
        "results": results,
        "track": track,
    }


col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Chain of thought (typed PlanGraph)")
    plan_box = st.empty()
    st.subheader("Map")
    map_box = st.empty()

with col2:
    st.subheader("Validator")
    val_box = st.empty()
    st.subheader("Cost / tokens")
    cost_box = st.empty()
    st.subheader("Result JSON")
    json_box = st.empty()


if not run and "last_out" not in st.session_state:
    st.info(
        "Pick a preset and click **Run**. The planner emits a typed PlanGraph "
        "(not free-form prose), the geometric kernel computes a Hägerstrand "
        "prism for each pair of anchors, the rendezvous finder intersects "
        "them with the DC-TGARD bi-directional pruning algorithm, and the "
        "kinematic validator gates every region returned to the user."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

state = _bootstrap()

# Compute only on a real Run click, then persist the result so the dashboard
# survives the reruns that Streamlit triggers (e.g. when st_folium loads). On
# those reruns `run` is False, so without this the page would fall through to
# the intro st.stop() and the whole dashboard would vanish.
if run:
    try:
        anchors = [
            Anchor(lat=float(a_lat), lon=float(a_lon), t=_parse_t(a_t)),
            Anchor(lat=float(b_lat), lon=float(b_lon), t=_parse_t(b_t)),
        ]
        if use_second_pair:
            anchors.extend([
                Anchor(lat=float(c_lat), lon=float(c_lon), t=_parse_t(c_t)),
                Anchor(lat=float(d_lat), lon=float(d_lon), t=_parse_t(d_t)),
            ])
        q = QueryIn(
            question=question,
            domain=domain,
            anchors=anchors,
            budget=Budget(max_tokens=int(max_tokens), max_tools=int(max_tools), max_seconds=float(max_seconds)),
        )
    except Exception as exc:
        st.error(f"Bad input: {exc}")
        st.stop()

    track = None
    if preset.get("track"):
        track = [
            {"lat": float(la), "lon": float(lo), "t": _parse_t(ti)}
            for (la, lo, ti) in preset["track"]
        ]

    with st.spinner("Running orchestrator..."):
        out = _event_loop().run_until_complete(_run(state, q, track=track))

    st.session_state["last_out"] = out
    st.session_state["last_q"] = q
else:
    out = st.session_state["last_out"]
    q = st.session_state["last_q"]


# Plan rendering
plan_box.code(_plan_to_mermaid(out["plan"]), language="mermaid")

# Map rendering: use folium if available, otherwise a Plotly scatter
try:
    import folium
    from streamlit_folium import st_folium

    centroid_lat = sum(a.lat for a in q.anchors) / len(q.anchors)
    centroid_lon = sum(a.lon for a in q.anchors) / len(q.anchors)
    m = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=6, tiles="cartodbpositron")
    # CircleMarker (vector) instead of folium.Marker, whose default PNG icon is
    # blocked by the Hugging Face Space sandbox and renders as a broken image.
    for idx, anchor in enumerate(q.anchors, start=1):
        folium.CircleMarker(
            [anchor.lat, anchor.lon],
            radius=6,
            color="#111827",
            weight=2,
            fill=True,
            fill_color="#ef4444",
            fill_opacity=0.9,
            tooltip=f"Anchor {idx} ({anchor.lat:.3f}, {anchor.lon:.3f})",
        ).add_to(m)

    colors = ["#3366ff", "#cc3300", "#7c3aed", "#0f766e"]
    for idx, prism in enumerate(out["prisms"], start=1):
        color = colors[(idx - 1) % len(colors)]
        folium.GeoJson(_ellipse_to_geojson(prism),
                       name=f"Prism {idx} (ellipse)",
                       style_function=lambda _f, c=color: {"color": c, "weight": 1, "fillOpacity": 0.15}
                       ).add_to(m)
        folium.GeoJson(_mobr_to_geojson(prism),
                       name=f"Prism {idx} (MOBR)",
                       style_function=lambda _f, c=color: {"color": c, "weight": 1, "dashArray": "4,3", "fill": False}
                       ).add_to(m)

    for r in out["regions"]:
        folium.GeoJson(r.polygon_geojson, name=f"Rendezvous ({r.method})",
                       style_function=lambda _f: {"color": "#22aa22", "weight": 2, "fillOpacity": 0.35}
                       ).add_to(m)
    for idx, gap in enumerate(out["gaps"], start=1):
        folium.GeoJson(gap.coverage_polygon_geojson, name=f"Gap {idx} coverage",
                       style_function=lambda _f: {"color": "#f59e0b", "weight": 2, "fillOpacity": 0.2}
                       ).add_to(m)
    # Reported track (the multi-point AIS path); the blackout shows as the long
    # straight segment, and the gap detector's amber coverage polygon sits over it.
    if out.get("track"):
        pts = [[p["lat"], p["lon"]] for p in out["track"]]
        folium.PolyLine(pts, color="#6b7280", weight=2, opacity=0.85,
                        tooltip="Reported track").add_to(m)
        for i, p in enumerate(pts, start=1):
            folium.CircleMarker(p, radius=3, color="#374151", weight=1, fill=True,
                                fill_color="#9ca3af", fill_opacity=0.95,
                                tooltip=f"ping {i}").add_to(m)

    # Frame the view on the computed geometry so the prism is visible without a
    # manual zoom-out.
    fit: list[list[float]] = []
    for prism in out["prisms"]:
        minx, miny, maxx, maxy = prism.ellipse_polygon().bounds  # lon, lat
        fit += [[miny, minx], [maxy, maxx]]
    if out.get("track"):
        fit += [[p["lat"], p["lon"]] for p in out["track"]]
    if fit:
        m.fit_bounds(fit, padding=(25, 25))

    folium.LayerControl(collapsed=False).add_to(m)
    with map_box:
        st_folium(m, width=720, height=480, returned_objects=[])
except Exception as exc:
    map_box.warning(f"Map renderer unavailable: {exc}")

# Validator
val_box.write({
    "regions_returned": len(out["regions"]),
    "gaps_detected": len(out["gaps"]),
    "method_in": sorted({r.method for r in out["regions"]}),
    "min_confidence": min((r.confidence for r in out["regions"]), default=None),
    "kinematic_pass": True,  # raised KinematicViolation otherwise
})

# Cost / tokens (planner only in this demo; orchestrator path is exercised in the full pipeline)
cost_box.write({
    "tokens_in":  state["planner"].last_tokens_in,
    "tokens_out": state["planner"].last_tokens_out,
    "cost_usd":   round(state["planner"].last_cost_usd, 6),
    "cache_hit":  state["planner"].last_cache_hit,
})

json_box.json({
    "question": q.question,
    "plan_rationale": out["plan"].rationale,
    "n_prisms": len(out["prisms"]),
    "n_gaps": len(out["gaps"]),
    "n_regions": len(out["regions"]),
    "gaps": [
        {
            "duration_s": g.duration_s,
            "distance_m": g.distance_m,
            "agm": g.abnormal_gap_measure,
        }
        for g in out["gaps"]
    ],
    "regions": [r.model_dump(mode="json") for r in out["regions"]],
})
