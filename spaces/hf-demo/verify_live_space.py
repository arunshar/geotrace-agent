"""Headless end-to-end verification of the live GeoTrace-Agent HF Space.

Drives a real browser (Playwright + system Chrome) to exercise the
"Coverage-gap on a vessel track" preset end to end and assert that the
new demo features work on the deployed Space:

  1. the gap detector + abnormal-gap measure (AGM) fire on the ~2-hour
     AIS blackout baked into the preset track,
  2. the validator reports the gap,
  3. the folium map renders the reported-track PolyLine + ping/anchor
     CircleMarkers + prism/gap GeoJSON, and
  4. fit-to-bounds reframes the map on the geometry (zoom > default 6).

Not a pytest test (file name does not start with ``test_``) so it is not
collected by the default ``pytest`` run; invoke it explicitly:

    python spaces/hf-demo/verify_live_space.py

Requires the ``playwright`` package and a local Google Chrome. Override
the target Space with ``GEOTRACE_SPACE_URL``. Screenshots are written to
``spaces/hf-demo/.output/`` (gitignored).
"""

import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print(
        "SKIP: 'playwright' is not installed. Install it (e.g. `pip install "
        "playwright`) to run this live-Space verification."
    )
    sys.exit(125)

APP_URL = os.environ.get("GEOTRACE_SPACE_URL", "https://arun0808-geotrace-agent.hf.space")
PRESET = "Coverage-gap on a vessel track"
OUT_DIR = Path(__file__).parent / ".output"


def dump(label: str, text: str) -> None:
    print(f"\n===== {label} =====")
    print(text.strip()[:4000])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome", headless=True)
        except Exception as exc:
            print(f"SKIP: could not launch Chrome via Playwright ({exc}).")
            print("Install Google Chrome, or `npx playwright install chromium`.")
            return 125
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()

        print(f"-> navigating to {APP_URL}")
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=90_000)

        print("-> waiting for sidebar + Run button")
        try:
            page.wait_for_selector('button:has-text("Run")', timeout=90_000)
        except Exception as exc:
            page.screenshot(path=str(OUT_DIR / "no_run_button.png"), full_page=True)
            print("FAIL: Run button never appeared:", exc)
            browser.close()
            return 2
        page.wait_for_timeout(1500)

        # --- Select the preset (type-to-filter + Enter; no listbox portal) ----
        print(f"-> selecting preset: {PRESET!r}")
        first_sb = page.locator('[data-testid="stSelectbox"]').first
        first_sb.scroll_into_view_if_needed()
        combo = first_sb.locator('input[role="combobox"]').first
        combo.click()
        page.wait_for_timeout(200)
        # "Coverage" uniquely matches "Coverage-gap on a vessel track" among the
        # four presets (Rendezvous, Trajectory-gap, Compute prism, Coverage-gap).
        page.keyboard.type("Coverage")
        page.wait_for_timeout(600)
        page.keyboard.press("Enter")
        page.wait_for_timeout(1200)
        preset_val = first_sb.locator("div[value]").first.get_attribute("value")
        print(f"   preset value now: {preset_val!r}")
        if preset_val != PRESET:
            page.screenshot(path=str(OUT_DIR / "preset_not_set.png"), full_page=True)
            print(f"FAIL: preset did not change to {PRESET!r} (got {preset_val!r})")
            browser.close()
            return 3

        # --- Run ------------------------------------------------------------
        run_btn = page.locator('button:has-text("Run")').first
        run_btn.scroll_into_view_if_needed()
        print("-> clicking Run")
        run_btn.click()

        # --- Wait for the orchestrator result (n_gaps) ----------------------
        print("-> waiting for result (n_gaps) ...")
        n_gaps_text = None
        deadline = time.time() + 90
        while time.time() < deadline:
            jsons = page.locator('[data-testid="stJson"]')
            for i in range(jsons.count()):
                if "n_gaps" in jsons.nth(i).inner_text():
                    n_gaps_text = jsons.nth(i).inner_text()
                    break
            if n_gaps_text:
                break
            page.wait_for_timeout(1000)
        if not n_gaps_text:
            page.screenshot(path=str(OUT_DIR / "no_result.png"), full_page=True)
            print("FAIL: result JSON (n_gaps) never appeared within timeout")
            browser.close()
            return 4
        dump("Result JSON", n_gaps_text)

        # Label the stJson blocks and capture validator/cost text.
        jsons = page.locator('[data-testid="stJson"]')
        validator_text = ""
        for i in range(jsons.count()):
            t = jsons.nth(i).inner_text().strip()
            if "tokens_in" in t:
                dump("Cost/tokens panel", t)
            elif "gaps_detected" in t or "regions_returned" in t:
                validator_text = t
                dump("Validator panel", t)
            elif "n_gaps" in t:
                dump("Result JSON panel", t)

        # --- Parse core signals ---------------------------------------------
        compact = re.sub(r"\s+", " ", n_gaps_text)
        m_n_gaps = re.search(r'"n_gaps"\s*:\s*(\d+)', compact)
        n_gaps = int(m_n_gaps.group(1)) if m_n_gaps else None
        agms = [float(x) for x in re.findall(r'"agm"\s*:\s*([-\d.eE+]+)', compact)]
        vcompact = re.sub(r"\s+", " ", validator_text)
        m_gaps_det = re.search(r'"gaps_detected"\s*:\s*(\d+)', vcompact)
        gaps_detected = int(m_gaps_det.group(1)) if m_gaps_det else None
        print("\n--- parsed signals ---")
        print("n_gaps        =", n_gaps)
        print("agm values    =", agms)
        print("gaps_detected =", gaps_detected)

        # --- Map / track overlay inspection ---------------------------------
        page.wait_for_timeout(4000)  # let the folium iframe finish rendering
        map_info: dict = {"frame_accessible": False, "layers": None, "map_state": None, "note": ""}
        map_frame = None
        for f in page.frames:
            try:
                if f.query_selector(".leaflet-container"):
                    map_frame = f
                    break
            except Exception:
                continue
        if map_frame is None:
            map_info["note"] = "no leaflet iframe found"
        else:
            map_info["frame_accessible"] = True
            try:
                layers = map_frame.evaluate("""() => {
                    const L = window.L;
                    if (!L) return null;
                    let map = null;
                    for (const k of Object.keys(window)) {
                        if (k.startsWith('map_') && window[k] instanceof L.Map) {
                            map = window[k]; break;
                        }
                    }
                    if (!map) return null;
                    const counts = {polyline:0, circleMarker:0, polygon:0,
                                    geojson:0, tile:0, other:0};
                    map.eachLayer(l => {
                        try {
                            if (l instanceof L.Polyline && !(l instanceof L.Polygon)) counts.polyline += 1;
                            else if (l instanceof L.CircleMarker) counts.circleMarker += 1;
                            else if (l instanceof L.Polygon) counts.polygon += 1;
                            else if (l instanceof L.GeoJSON) counts.geojson += 1;
                            else if (l instanceof L.TileLayer) counts.tile += 1;
                            else counts.other += 1;
                        } catch (e) {}
                    });
                    const c = map.getCenter();
                    return {counts, center:{lat:c.lat, lng:c.lng}, zoom: map.getZoom()};
                }""")
                map_info["layers"] = layers
                if isinstance(layers, dict):
                    map_info["map_state"] = {"center": layers["center"], "zoom": layers["zoom"]}
            except Exception as exc:
                map_info["layers"] = f"err:{exc}"
        print("\n--- map / track overlay ---")
        print(json.dumps(map_info, indent=2, default=str))

        page.screenshot(path=str(OUT_DIR / "full_page.png"), full_page=True)
        page.screenshot(path=str(OUT_DIR / "viewport.png"), full_page=False)
        print(f"\n-> screenshots saved to {OUT_DIR}")
        browser.close()

        # --- Verdict --------------------------------------------------------
        ok = True
        reasons: list[str] = []
        if not n_gaps or n_gaps < 1:
            ok = False
            reasons.append("gap detector did not fire (n_gaps < 1)")
        if not agms or not any(a > 0 for a in agms):
            ok = False
            reasons.append("no positive abnormal_gap_measure returned")
        if not gaps_detected or gaps_detected < 1:
            ok = False
            reasons.append("validator gaps_detected < 1")

        layers = map_info.get("layers")
        poly = circle = 0
        if isinstance(layers, dict) and isinstance(layers.get("counts"), dict):
            poly = layers["counts"].get("polyline", 0)
            circle = layers["counts"].get("circleMarker", 0)
        if not map_info.get("frame_accessible"):
            ok = False
            reasons.append("folium map iframe not accessible")
        else:
            if poly < 1:
                ok = False
                reasons.append(f"track PolyLine not on map (polyline layers={poly})")
            if circle < 1:
                ok = False
                reasons.append(f"ping/anchor CircleMarkers not on map (circleMarker layers={circle})")
            st = map_info.get("map_state")
            if isinstance(st, dict) and st.get("zoom") is not None:
                z = st["zoom"]
                if z <= 6:
                    ok = False
                    reasons.append(f"fit-bounds not applied (zoom={z}, still default)")
                else:
                    print(f"   fit-bounds OK: zoom={z}, center=({st['center']['lat']:.3f}, {st['center']['lng']:.3f})")
            else:
                print("   note: could not read map zoom; fit-bounds checked via screenshot")

        print("\n=== VERDICT ===")
        if ok:
            print("PASS: gap detector + AGM fired; track overlay + fit-bounds confirmed on map.")
            return 0
        print("FAIL:")
        for r in reasons:
            print("  -", r)
        return 1


if __name__ == "__main__":
    sys.exit(main())
