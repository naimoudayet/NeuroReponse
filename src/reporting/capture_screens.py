"""Drive the running Streamlit app with Playwright and save PNG screenshots.

Usage:
    1. Make sure DB is seeded + model is trained:
         python -m src.data.seeder
    2. Start the app on port 8765:
         streamlit run src/app/main.py --server.headless true --server.port 8765
    3. Run:
         python -m src.reporting.capture_screens
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

BASE_URL = "http://localhost:8765"
OUT_DIR = Path("docs/screenshots")
VIEWPORT = {"width": 1400, "height": 900}


def _wait_for_app(page: Page, max_wait: float = 30.0) -> None:
    """Wait until Streamlit finishes its initial render."""
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


def capture(page: Page) -> None:
    # 1. Home
    page.goto(f"{BASE_URL}/", wait_until="networkidle")
    _wait_for_app(page)
    _shot(page, "01_home.png")

    # 2. Patients
    page.goto(f"{BASE_URL}/Patients", wait_until="networkidle")
    _wait_for_app(page)
    time.sleep(1.5)
    _shot(page, "02_patients.png")

    # 3. Sessions
    page.goto(f"{BASE_URL}/Sessions", wait_until="networkidle")
    _wait_for_app(page)
    time.sleep(1.5)
    _shot(page, "03_sessions.png")

    # 4. Training
    page.goto(f"{BASE_URL}/Training", wait_until="networkidle")
    _wait_for_app(page)
    time.sleep(1.5)
    _shot(page, "04_training.png")

    # 5. Predictions
    page.goto(f"{BASE_URL}/Predictions", wait_until="networkidle")
    _wait_for_app(page)
    time.sleep(2.0)
    _shot(page, "05_predictions.png")


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
        page = ctx.new_page()
        try:
            capture(page)
        finally:
            ctx.close()
            browser.close()


if __name__ == "__main__":
    main()
