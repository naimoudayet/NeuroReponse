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
def test_a_hidden_cohort_falls_back_instead_of_crashing(page, monkeypatch):
    """A stored selection naming a cohort the app no longer offers must not raise.

    A session resumed after the sidebar roster changed used to hit
    `list.index(x): x not in list` on *every* page. All three cohorts are
    selectable again now that the app serves the whole 2x2, so the roster is
    narrowed here rather than relying on one happening to be hidden — otherwise
    this test silently stops exercising the fallback it exists for.
    """
    from src.app import utils

    _requires_db(DataSource.SIMULE_SEQ)
    monkeypatch.setattr(utils, "VISIBLE", (DataSource.SIMULE_SEQ,))
    at = _run(page, data_source=DataSource.TDBRAIN.value)
    assert not at.exception, f"{page.name}: {at.exception}"


@pytest.mark.parametrize("protocol, expected", [(1, 44), (2, 88)])
def test_patient_list_is_filtered_to_the_selected_protocol(protocol, expected):
    """A model fitted on one rTMS arm must not be offered the other's patients.

    Nothing downstream would catch it: the feature vector is the right shape for
    either arm, so the page would predict across treatments and report it as a
    normal result.
    """
    _requires_db(DataSource.TDBRAIN)
    from src.app.utils import get_repository, list_patient_ids

    repo = get_repository(DataSource.TDBRAIN)
    if len(list_patient_ids(repo)) != 132:
        pytest.skip("real cohort seeded with a different patient count")
    assert len(list_patient_ids(repo, protocol)) == expected


def test_predictions_renders_the_article_regression_variant():
    """End-to-end on the new axes: protocol -> regression checkpoint -> BDI points."""
    _requires_db(DataSource.TDBRAIN)
    cfg = SOURCES[DataSource.TDBRAIN]
    variant = next(
        m for m in cfg.regression_models
        if m.protocol == 1 and m.uses_signals
    )
    if not variant.model.exists():
        pytest.skip(f"{variant.model} not trained on this machine")

    at = _run(
        APP / "pages" / "4_Predictions.py",
        data_source=DataSource.TDBRAIN.value,
        **{
            f"protocole:{cfg.label}": 1,
            f"model_choice:{cfg.label}": variant.key,
        },
    )
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    # The regression head reports points recovered, never a probability.
    assert any("Réduction BDI-II prédite" in label for label in labels)
    assert not any("Probabilité" in label for label in labels)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders_on_the_article_regression_arm(page):
    """Every page must survive the regression axes, not just Predictions."""
    _requires_db(DataSource.TDBRAIN)
    cfg = SOURCES[DataSource.TDBRAIN]
    at = _run(
        page,
        data_source=DataSource.TDBRAIN.value,
        **{f"protocole:{cfg.label}": 1},
    )
    assert not at.exception, f"{page.name}: {at.exception}"


def test_predictions_runs_the_multimodal_variant():
    """The end-to-end path: variant -> contract -> clinical block -> model."""
    _requires_db(DataSource.TDBRAIN)
    cfg = SOURCES[DataSource.TDBRAIN]
    multi = next(
        m for m in cfg.regression_models
        if m.protocol == 1 and set(m.modalities) >= {"rtms", "eeg", "ecg"}
    )
    if not multi.model.exists():
        pytest.skip(f"{multi.model} not trained on this machine")

    at = _run(
        APP / "pages" / "4_Predictions.py",
        data_source=DataSource.TDBRAIN.value,
        **{f"protocole:{cfg.label}": 1, f"model_choice:{cfg.label}": multi.key},
    )
    assert not at.exception, at.exception
    # A refusal to predict is reported as an error box, never a silent blank.
    assert not at.error, [e.value for e in at.error]


def test_clinical_variant_predicts_without_reading_a_recording():
    _requires_db(DataSource.TDBRAIN)
    cfg = SOURCES[DataSource.TDBRAIN]
    clinical = next(
        m for m in cfg.regression_models
        if m.protocol == 1 and not m.uses_signals
    )
    if not clinical.model.exists():
        pytest.skip(f"{clinical.model} not trained on this machine")

    at = _run(
        APP / "pages" / "4_Predictions.py",
        data_source=DataSource.TDBRAIN.value,
        **{f"protocole:{cfg.label}": 1, f"model_choice:{cfg.label}": clinical.key},
    )
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]


# --------------------------------------------------------------------------- #
# The 2x2: all four comparison checkpoints must be *reachable*, not just trained.
# --------------------------------------------------------------------------- #

def test_every_2x2_variant_is_reachable_from_the_sidebar():
    """Trained is not the same as offered.

    All four comparison checkpoints existed on disk, with their sidecars, while
    the app served none of them: the matched simulated cohort was absent from
    `VISIBLE`, and the real cohort was pinned to the regression head, so its
    pooled classification pair could never surface. Nothing failed — the models
    simply appeared only as recorded numbers on the comparison page.
    """
    from src.app.utils import VISIBLE, available_heads, available_models
    from src.models.variants import ORDERED, Task

    reachable: set[str] = set()
    for source in VISIBLE:
        cfg = SOURCES[source]
        for head in available_heads(cfg):
            arms = (None,) if head is Task.CLASSIFICATION else (1, 2)
            for arm in arms:
                reachable |= {m.key for m in available_models(source, head, arm)}

    missing = [v.value for v in ORDERED if v.value not in reachable]
    assert not missing, f"variantes 2x2 injoignables depuis la barre latérale : {missing}"


def test_the_binary_head_pools_both_arms():
    """Selecting the binary head must clear the protocol filter.

    The 2x2 checkpoints carry ``protocol=None``. If ``current_protocol`` kept
    returning an arm, ``available_models`` would match nothing and
    ``model_choice`` would fall back to ``cfg.models[0]`` — serving a checkpoint
    the sidebar never selected, against a patient list filtered to one arm, with
    nothing on screen looking wrong.

    ``main.py`` renames its metric when an arm is active, so the label is a
    faithful witness of which branch ran.
    """
    from src.models.variants import Task

    _requires_db(DataSource.TDBRAIN)
    cfg = SOURCES[DataSource.TDBRAIN]
    at = _run(
        APP / "main.py",
        data_source=DataSource.TDBRAIN.value,
        **{
            f"objectif:{cfg.label}": Task.CLASSIFICATION.value,
            f"protocole:{cfg.label}": 1,      # stale arm from the regression head
        },
    )
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert "Patients en base" in labels, labels
    assert "Patients (bras sélectionné)" not in labels, labels


def test_predictions_serves_the_pooled_2x2_checkpoint():
    """End-to-end on the binary head: a probability, never BDI-II points."""
    from src.models.variants import Task

    _requires_db(DataSource.TDBRAIN)
    cfg = SOURCES[DataSource.TDBRAIN]
    variant = next(
        m for m in cfg.models if set(m.modalities) >= {"rtms", "eeg", "ecg"}
    )
    if not variant.model.exists():
        pytest.skip(f"{variant.model} not trained on this machine")

    at = _run(
        APP / "pages" / "4_Predictions.py",
        data_source=DataSource.TDBRAIN.value,
        **{
            f"objectif:{cfg.label}": Task.CLASSIFICATION.value,
            f"model_choice:{cfg.label}": variant.key,
        },
    )
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    labels = [m.label for m in at.metric]
    assert any("Probabilité" in label for label in labels), labels
    assert not any("Réduction BDI-II prédite" in label for label in labels), labels


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders_on_the_binary_head_of_the_real_cohort(page):
    """Every page must survive the head the app previously could not reach."""
    from src.models.variants import Task

    _requires_db(DataSource.TDBRAIN)
    cfg = SOURCES[DataSource.TDBRAIN]
    at = _run(
        page,
        data_source=DataSource.TDBRAIN.value,
        **{f"objectif:{cfg.label}": Task.CLASSIFICATION.value},
    )
    assert not at.exception, f"{page.name}: {at.exception}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders_on_the_matched_simulated_cohort(page):
    """The simulated half of the 2x2, newly selectable."""
    _requires_db(DataSource.SIMULE)
    at = _run(page, data_source=DataSource.SIMULE.value)
    assert not at.exception, f"{page.name}: {at.exception}"


def test_an_unknown_stored_head_falls_back():
    """A stored head the cohort cannot serve must not raise."""
    _requires_db(DataSource.TDBRAIN)
    cfg = SOURCES[DataSource.TDBRAIN]
    at = _run(
        APP / "main.py",
        data_source=DataSource.TDBRAIN.value,
        **{f"objectif:{cfg.label}": "regression_quantile_inexistante"},
    )
    assert not at.exception, at.exception
