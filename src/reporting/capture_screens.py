"""Drive the running Streamlit app with Playwright and save PNG screenshots.

Usage:
    1. Make sure the databases are seeded and the models trained:
         python -m src.data.seeder                       # simulated
         python -m src.models.train_tdbrain --root <TDBRAIN root> \
             --seed-db recherche_tdbrain.sqlite3         # real
    2. Start the app on port 8765:
         streamlit run src/app/main.py --server.headless true --server.port 8765
    3. Run:
         python -m src.reporting.capture_screens               # both cohorts
         python -m src.reporting.capture_screens --source tdbrain

Navigation deliberately clicks the sidebar links rather than calling ``goto`` per
page: a full page load starts a **new Streamlit session**, which would reset the
data-source selector back to the simulated cohort mid-capture.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

from src.app.utils import SOURCES

BASE_URL = "http://localhost:8765"
OUT_DIR = Path("docs/screenshots")
VIEWPORT = {"width": 1400, "height": 900}

# (sidebar link label, screenshot suffix) — order matches the guide's walkthrough.
PAGES: list[tuple[str, str]] = [
    ("Patients", "patients"),
    ("Sessions", "sessions"),
    ("Training", "training"),
    ("Predictions", "predictions"),
    ("Suivi", "suivi"),
    ("Boucle clinique", "boucle"),
    ("Comparaison", "comparaison"),
]

# Read from the app's own catalogue rather than restated here: these labels are
# clicked by exact text, so a wording change in the sidebar would otherwise make
# the capture silently screenshot the wrong cohort.
SOURCE_LABELS = {s.value: cfg.label for s, cfg in SOURCES.items()}

# Which cohort gets which screenshot prefix in the guide.
PREFIXES = {"simule_seq": "", "simule": "sim_", "tdbrain": "td_"}


def _wait_for_app(page: Page, max_wait: float = 60.0) -> None:
    """Wait until Streamlit finishes rendering (no running status widget)."""
    end = time.time() + max_wait
    while time.time() < end:
        try:
            page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=2_000)
            page.wait_for_function(
                "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')"
                " || document.querySelector('[data-testid=\"stStatusWidget\"]').innerText.trim() === ''",
                timeout=3_000,
            )
            return
        except Exception:
            time.sleep(0.5)
    print(f"  warning: still waiting for {page.url}", file=sys.stderr)


def _shot(page: Page, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    page.screenshot(path=str(path), full_page=True)
    print(f"  wrote {path}")
    return path


def _select_source(page: Page, source: str) -> None:
    """Click the sidebar cohort radio and wait for the rerun."""
    label = SOURCE_LABELS[source]
    page.get_by_text(label, exact=True).first.click()
    _wait_for_app(page)
    time.sleep(1.5)


def _run_cross_validation(page: Page, timeout_s: float = 600.0) -> bool:
    """Click 'Lancer la validation croisée' and wait for the metrics to appear.

    A guide screenshot of an empty form teaches nothing; this captures the page
    with real cross-validation output. Returns False if it did not finish in time.
    """
    try:
        page.get_by_role("button", name="Lancer la validation croisée patient-wise").click()
    except Exception as exc:
        print(f"  note: could not start cross-validation ({exc})", file=sys.stderr)
        return False
    end = time.time() + timeout_s
    while time.time() < end:
        _wait_for_app(page, max_wait=15.0)
        if page.locator("[data-testid='stMetric']").count() >= 3:
            time.sleep(2.0)
            return True
        time.sleep(2.0)
    print("  note: cross-validation did not finish in time", file=sys.stderr)
    return False


def _select_ecg_channel(page: Page) -> bool:
    """On the Sessions page, switch the channel picker to the ECG lead.

    The tachogram view is the visual proof that the autonomic modality is stored,
    so the guide needs it. The dropdown is virtualised (the ECG row is the 27th),
    hence typing to filter rather than scrolling.
    """
    try:
        boxes = page.locator("div[data-baseweb='select']")
        if boxes.count() < 2:
            return False
        boxes.nth(boxes.count() - 1).click()
        time.sleep(1.0)
        page.keyboard.type("Erbs")
        time.sleep(1.2)
        opts = page.locator("li[role='option']")
        if not opts.count():
            page.keyboard.press("Escape")
            return False
        opts.first.click()
        _wait_for_app(page)
        time.sleep(2.0)
        return True
    except Exception as exc:  # capture is best-effort; never abort the run
        print(f"  note: could not select the ECG channel ({exc})", file=sys.stderr)
        return False


def capture(page: Page, source: str, prefix: str, train: bool = True) -> None:
    print(f"\n== capturing cohort '{source}' (prefix {prefix}) ==")
    page.goto(f"{BASE_URL}/", wait_until="networkidle")
    _wait_for_app(page)
    _select_source(page, source)
    _shot(page, f"{prefix}01_home.png")

    for idx, (link, suffix) in enumerate(PAGES, start=2):
        page.get_by_role("link", name=link).first.click()
        _wait_for_app(page)
        time.sleep(2.0)
        if suffix == "training" and train:
            _shot(page, f"{prefix}04_training_form.png")
            if _run_cross_validation(page):
                _shot(page, f"{prefix}04_training.png")
                continue
        _shot(page, f"{prefix}{idx:02d}_{suffix}.png")
        # Extra shot: the ECG tachogram only exists on the real cohort.
        if suffix == "sessions" and source == "tdbrain" and _select_ecg_channel(page):
            _shot(page, f"{prefix}{idx:02d}b_sessions_ecg.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Screenshot the Streamlit app.")
    ap.add_argument("--source", choices=(*SOURCE_LABELS, "both"), default="both")
    ap.add_argument("--no-train", action="store_true",
                    help="skip running cross-validation before the Training shot")
    args = ap.parse_args()

    targets = (
        [(k, PREFIXES[k]) for k in SOURCE_LABELS]
        if args.source == "both"
        else [(args.source, PREFIXES[args.source])]
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
        page = ctx.new_page()
        try:
            for source, prefix in targets:
                capture(page, source, prefix, train=not args.no_train)
        finally:
            ctx.close()
            browser.close()


if __name__ == "__main__":
    main()
