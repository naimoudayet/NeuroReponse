"""Tests for the evaluation charts and the notebook builder.

The charts are how the four models get read, so the logic that decides *what the
reader is told* — the verdict classifier — is tested directly rather than left to
visual inspection. The drawing functions are smoke-tested for shape and failure
modes; their appearance is checked by eye when the notebooks are regenerated.
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.modalities import build_features  # noqa: E402
from src.data.simulator_matched import (  # noqa: E402
    MatchedSimConfig,
    simulate_matched,
)
from src.models.lstm import LSTMConfig  # noqa: E402
from src.models.train import TrainConfig, cross_validate  # noqa: E402
from src.reporting import model_charts as mc  # noqa: E402


@pytest.fixture(scope="module")
def cv_bundle():
    ds = simulate_matched(
        MatchedSimConfig(n_patients=24, n_epochs=3, window=256, seed=2)
    )
    x, y, groups, names = build_features(ds, modalities=("rtms", "eeg", "ecg"))
    cv = cross_validate(
        x, y.astype(np.float32), groups,
        lstm_cfg=LSTMConfig(input_size=x.shape[-1]),
        train_cfg=TrainConfig(epochs=2, batch_size=8), n_splits=3,
    )
    return cv, x, y, groups, names


# --------------------------------------------------------------------------- #
# Out-of-fold predictions
# --------------------------------------------------------------------------- #


def test_every_patient_is_scored_exactly_once(cv_bundle):
    """GroupKFold validates each patient once; the pooled OOF set must reflect
    that, or the ROC would double-count some patients and omit others."""
    cv, _x, y, _g, _n = cv_bundle
    y_true, y_proba = cv.out_of_fold(y)
    assert len(y_true) == len(y) == len(y_proba)
    assert set(np.concatenate([f.val_idx for f in cv.folds])) == set(range(len(y)))
    assert ((y_proba >= 0.0) & (y_proba <= 1.0)).all()


def test_out_of_fold_labels_line_up_with_their_probabilities(cv_bundle):
    """Returned in **original patient order**, with each score still on its own patient.

    Order matters beyond tidiness: callers correlate these against per-patient
    covariates (`bdi_pre`, age) held in the dataset's order. Returning fold order
    silently paired mismatched vectors and turned a baseline of r = 0.500 into
    r = 0.090.
    """
    cv, _x, y, _g, _n = cv_bundle
    y_true, y_proba = cv.out_of_fold(y)

    np.testing.assert_array_equal(y_true, np.asarray(y))
    # Each fold's stored probabilities must sit at that fold's patient indices.
    for fold in cv.folds:
        np.testing.assert_allclose(y_proba[fold.val_idx], fold.val_proba)


def test_out_of_fold_rejects_a_result_without_stored_probabilities():
    from src.models.train import CVResult, FoldResult

    stale = CVResult(folds=[FoldResult(
        fold=0, train_idx=np.array([0]), val_idx=np.array([1]),
        train_losses=[], val_losses=[], accuracy=0.5, auc=0.5, f1=0.5, best_epoch=0,
    )])
    with pytest.raises(ValueError, match="val_proba"):
        stale.out_of_fold(np.array([0, 1]))


# --------------------------------------------------------------------------- #
# The verdict classifier — what the reader is actually told
# --------------------------------------------------------------------------- #


def test_verdict_calls_a_base_rate_predictor_what_it_is():
    status, sentence = mc.verdict(auc=0.52, auc_std=0.12, accuracy=0.63, base_rate=0.63)
    assert status == "critical"
    assert "hasard" in sentence


def test_verdict_recognises_a_genuinely_useful_model():
    status, _ = mc.verdict(auc=0.85, auc_std=0.05, accuracy=0.80, base_rate=0.63)
    assert status == "good"


def test_verdict_flags_a_mean_whose_spread_covers_chance():
    """The exact situation of this project: above 0.5 on average, but the fold
    spread reaches back through it, so nothing is established."""
    status, sentence = mc.verdict(auc=0.574, auc_std=0.111, accuracy=0.70, base_rate=0.63)
    assert status == "warning"
    assert "dispersion" in sentence


def test_verdict_marks_a_single_satisfied_criterion_as_marginal():
    status, _ = mc.verdict(auc=0.80, auc_std=0.02, accuracy=0.63, base_rate=0.63)
    assert status == "serious"


@pytest.mark.parametrize("auc,std,acc,base", [
    (0.50, 0.10, 0.63, 0.63), (0.99, 0.01, 0.95, 0.63),
    (0.30, 0.05, 0.40, 0.63), (0.63, 0.20, 0.63, 0.63),
])
def test_verdict_always_returns_a_known_status(auc, std, acc, base):
    status, sentence = mc.verdict(auc, std, acc, base)
    assert status in mc.STATUS
    assert sentence


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


def test_full_report_has_six_panels(cv_bundle):
    cv, x, y, groups, names = cv_bundle
    fig = mc.model_report("Test", cv, y, x, groups, names, base_rate=0.63)
    assert len(fig.axes) == 6
    plt.close(fig)


def test_roc_returns_the_pooled_auc(cv_bundle):
    cv, _x, y, _g, _n = cv_bundle
    y_true, y_proba = cv.out_of_fold(y)
    fig, ax = plt.subplots()
    auc = mc.plot_roc(ax, y_true, y_proba)
    assert 0.0 <= auc <= 1.0
    plt.close(fig)


def test_calibration_survives_too_few_patients_per_bin():
    """Small cohorts can make calibration_curve raise; the panel must degrade to
    a message rather than take the whole report down."""
    fig, ax = plt.subplots()
    mc.plot_calibration(ax, np.array([0, 1, 1]), np.array([0.4, 0.6, 0.6]), n_bins=10)
    plt.close(fig)


def test_confusion_matrix_counts_match_the_predictions(cv_bundle):
    cv, _x, y, _g, _n = cv_bundle
    y_true, y_proba = cv.out_of_fold(y)
    fig, ax = plt.subplots()
    mc.plot_confusion(ax, y_true, y_proba)
    texts = [t.get_text() for t in ax.texts if t.get_text().isdigit()]
    assert sum(int(t) for t in texts) == len(y_true)
    plt.close(fig)


def test_permutation_importance_is_sorted_and_capped(cv_bundle):
    _cv, x, y, groups, names = cv_bundle
    feats, vals = mc.permutation_importance(x, y, groups, names, n_repeats=2, top_n=5)
    assert len(feats) == len(vals) == 5
    assert vals == sorted(vals, reverse=True)
    assert set(feats) <= set(names)


def test_palette_is_the_validated_set():
    """These three hues passed the colour-vision validator together; swapping one
    in isolation would silently break that guarantee."""
    assert mc.SERIES == ("#2a78d6", "#eb6834", "#1baf7a")
    assert set(mc.STATUS) == {"good", "warning", "serious", "critical"}


# --------------------------------------------------------------------------- #
# Notebook builder
# --------------------------------------------------------------------------- #


def test_builder_emits_four_valid_notebooks(tmp_path, monkeypatch):
    from src.reporting import build_notebooks as bn

    monkeypatch.setattr(bn, "NOTEBOOK_DIR", tmp_path)
    paths = bn.build_all()

    assert len(paths) == 4
    numbers = [p.name[:2] for p in paths]
    assert numbers == ["05", "06", "07", "08"]        # the app's ordering

    for path in paths:
        nb = json.loads(path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        assert nb["cells"]
        # Every cell needs an id: nbformat will make its absence a hard error.
        ids = [c["id"] for c in nb["cells"]]
        assert len(ids) == len(set(ids))
        assert any(c["cell_type"] == "code" for c in nb["cells"])


def test_only_the_simulated_multimodal_notebook_runs_the_positive_control(
    tmp_path, monkeypatch
):
    """The effect sweep needs a generator with an effect knob, so it belongs to
    the simulated cohort only — the real one has no such control."""
    from src.reporting import build_notebooks as bn

    monkeypatch.setattr(bn, "NOTEBOOK_DIR", tmp_path)
    has_control = {}
    for path in bn.build_all():
        text = path.read_text(encoding="utf-8")
        has_control[path.name[:2]] = "Contrôle positif" in text

    assert has_control == {"05": False, "06": False, "07": True, "08": False}
