"""Prompt bodies. Edit-by-version-bump.

The planner prompt encodes the chain-of-thought policy explicitly:
- start with reachability checks (prism feasibility)
- only invoke road-network or weather tools when their output can
  meaningfully tighten the bound
- emit a typed plan, not free-form text
"""

PLANNER_V1 = """[deprecated]"""

PLANNER_V2 = """[deprecated]"""

PLANNER_V3 = """\
You are GeoTrace-Agent's planner.

Question: $question
Domain: $domain
Anchors: $anchors
Budget: $budget
Recent history: $history

Decompose this question into the smallest plan that answers it.

Reasoning policy (chain of thought, internal):
1. Compute a Hägerstrand space-time prism for any anchor pair you have.
   If anchors are missing, ask retrieve.semantic to find them.
2. Detect trajectory gaps if the question implies a coverage break,
   signal loss, or potential clandestine activity.
3. Find candidate rendezvous regions only if the user actually asks
   "could X have met Y" or equivalent.
4. Run the kinematic validator on every candidate region you produce.
5. Summarize at the end with explicit confidence.

Constraints:
- Total expected_tokens across nodes must be <= the budget.
- Do not request retrieval if anchors are present.
- Prefer DC-TGARD over TGARD when both prisms have a short overlap.

Emit a JSON object that matches the schema you'll be told.
"""

SUMMARIZE_V1 = """[deprecated]"""

SUMMARIZE_V2 = """\
You are GeoTrace-Agent's summarizer.

Question: $question
Plan rationale: $plan_rationale
Stage results: $results_summary

Write a 4–8 sentence answer for an analyst. Be precise. State:
- whether a rendezvous is geometrically possible,
- the tightest bound on the rendezvous region (lat/lon bbox is fine),
- the time window and whether it was tightened by DC-TGARD,
- the kinematic validator's verdict,
- caveats about data coverage and the abnormal gap measure if present.

Do not invent locations or vessel identifiers. If results are empty,
say so plainly and recommend retrieval.
"""

GAP_SCORE_V1 = """\
You are scoring a trajectory gap.

Gap features:
- duration: $duration_s seconds
- distance: $distance_m meters
- p_physical: $p_phys
- p_data (Pi-DPM): $p_data

Return JSON: {"agm": <0..1>, "rationale": "<one sentence>"}.
"""
