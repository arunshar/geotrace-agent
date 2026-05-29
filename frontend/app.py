"""Streamlit ops console.

Two views:
1. Query console. Submit a question, see the typed plan, the regions
   on a Folium map, the per-stage cost breakdown, and a button to send
   the trace to HITL review.
2. HITL queue. Reviewer picks an item, accepts/rejects, optionally
   marks as a DPO preference pair (preferred vs alternative region).
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("GEOTRACE_API_URL", "http://localhost:8000")


def main() -> None:
    st.set_page_config(page_title="GeoTrace-Agent", layout="wide")
    st.title("GeoTrace-Agent")
    tab_query, tab_hitl = st.tabs(["Query", "HITL queue"])
    with tab_query:
        question = st.text_area("Question", "Could VESSEL-1234 have rendezvoused with VESSEL-9876 between 06:00Z and 12:00Z?")
        if st.button("Run"):
            with httpx.Client(timeout=60.0) as c:
                r = c.post(f"{API_URL}/v1/query", json={"question": question, "domain": "vessel"})
                r.raise_for_status()
                out = r.json()
            st.json(out)
    with tab_hitl:
        st.write("HITL queue lives in Postgres; this view is hot-loaded by the seed script.")


if __name__ == "__main__":  # pragma: no cover
    main()
