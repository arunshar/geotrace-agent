---
title: GeoTrace-Agent
emoji: 🛰️
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.58.0"
app_file: streamlit_app.py
pinned: true
license: mit
short_description: Space-time prism and rendezvous reasoning agent demo
---

# GeoTrace-Agent — Hugging Face Space

A standalone Streamlit demo of [GeoTrace-Agent](https://github.com/arunshar/geotrace-agent), the production multi-agent framework for spatiotemporal reasoning described in the [NeurIPS-style preprint](https://github.com/arunshar/geotrace-agent/blob/main/paper/geotrace_agent_neurips.pdf).

This Space runs on the free CPU tier. It does NOT require Postgres, Redis, Chroma, or any external service. It can run in two modes:

| Mode | What you get | How |
|---|---|---|
| **Offline stub** (default) | Deterministic planner stub; full geometric kernel (prism, ellipse, MOBR, DRM); STAGD + DC-TGARD agents; kinematic validator; chain-of-thought visualization | No setup. |
| **Live planner** | Anthropic Claude Sonnet 4.6 generates the typed PlanGraph; everything else as above | Add `ANTHROPIC_API_KEY` as a Space secret. |

## What you can try

1. Pick one of the preset queries (rendezvous, gap audit, prism only) or type your own.
2. Adjust the anchors and the budget (max tokens / max tools / max seconds). The rendezvous preset exposes two anchor pairs, one per vessel.
3. Click **Run**.
4. Inspect:
   - the typed `PlanGraph` (rendered as a Mermaid DAG of typed nodes);
   - the prism on a Folium map with the inscribed geo-ellipse and MOBR;
   - the candidate rendezvous regions (TGARD vs DC-TGARD);
   - the kinematic validator's verdict;
   - the per-stage cost / token panel and the full JSON response.

## Cite

```bibtex
@article{sharma2026geotrace,
  title  = {{GeoTrace-Agent}: A Production Multi-Agent Framework for Spatiotemporal Reasoning with Hägerstrand Space-Time Prisms},
  author = {Sharma, Arun},
  year   = {2026}
}
```
