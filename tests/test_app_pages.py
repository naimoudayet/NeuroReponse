"""Smoke tests: every page renders without raising.

Streamlit pages are scripts, not functions — nothing imports them, so a rename in
``utils`` or a stale keyword argument stays invisible until someone clicks the
page. ``AppTest`` executes them the way the browser would and surfaces the
exception here instead.

These are deliberately shallow: they assert the page *runs* and reaches its
widgets, not what it displays. The numerical guarantees live in
``test_inference_clinical.py`` and ``test_variants.py``.

Pages that need a seeded database are skipped when it is absent, so the suite
still passes on a machine that has never run the seeders.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.app.utils import SOURCES, DataSource

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = Path("src/app")
PAGES = [
    APP / "main.py",
    *sorted((APP / "pages").glob("[0-9]_*.py")),
]

# Loading a montage cohort out of SQLite is genuinely slow (132 patients x 8
# epochs x 26 channels), so the pages that do it get a generous budget.
TIMEOUT = 300


def _requires_db(source: DataSource) -> None:
    db = SOURCES[source].db
    if not db.exists():
        pytest.skip(f"{db} not seeded on this machine")


def _run(path: Path, **state):
    at = AppTest.from_file(str(path), default_timeout=TIMEOUT)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders_on_the_default_cohort(page):
    """Default state is the legacy sequential cohort."""
    _requires_db(DataSource.SIMULE_SEQ)
    at = _run(page)
    assert not at.exception, f"{page.name}: {at.exception}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders_on_the_matched_cohort(page):
    """The matched cohort is the one the 2x2's simulated variants were fit on."""
    _requires_db(DataSource.SIMULE)
    at = _run(page, data_source=DataSource.SIMULE.value)
    assert not at.exception, f"{page.name}: {at.exception}"


def test_predictions_runs_the_multimodal_variant():
    """The end-to-end path: variant -> contract -> clinical block -> model."""
    _requires_db(DataSource.SIMULE)
    cfg = SOURCES[DataSource.SIMULE]
    multi = next(m for m in cfg.models if m.uses_signals)
    if not multi.model.exists():
        pytest.skip(f"{multi.model} not trained on this machine")

    at = _run(
        APP / "pages" / "4_Predictions.py",
        data_source=DataSource.SIMULE.value,
        **{f"model_choice:{cfg.label}": multi.key},
    )
    assert not at.exception, at.exception
    # A refusal to predict is reported as an error box, never a silent blank.
    assert not at.error, [e.value for e in at.error]


def test_clinical_variant_predicts_without_reading_a_recording():
    _requires_db(DataSource.SIMULE)
    cfg = SOURCES[DataSource.SIMULE]
    clinical = next(m for m in cfg.models if not m.uses_signals)
    if not clinical.model.exists():
        pytest.skip(f"{clinical.model} not trained on this machine")

    at = _run(
        APP / "pages" / "4_Predictions.py",
        data_source=DataSource.SIMULE.value,
        **{f"model_choice:{cfg.label}": clinical.key},
    )
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
