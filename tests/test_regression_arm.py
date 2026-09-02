"""Tests for the article-aligned regression arm.

Three things have to hold for this arm to mean anything:

1. **The two heads cannot be confused.** A regression checkpoint emits BDI-II
   *points*; pushing those through a sigmoid produces a perfectly normal-looking
   probability curve on every page in the app. The model must refuse instead.
2. **Protocol filtering is real.** The two rTMS arms are different treatments,
   and a model fitted on one must never be offered the other's patients.
3. **The trivial baseline is remembered.** `delta_bdi` is coupled to baseline
   severity; the bar is pinned here so it cannot quietly stop being checked.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.modalities import build_features, protocol_mask, target_values
from src.models.lstm import CLASSIFICATION, REGRESSION, LSTMConfig, ResponseLSTM
from src.models.metrics import pearson_r
from src.models.train import TrainConfig, cross_validate


# --------------------------------------------------------------------------- #
# 1. The heads refuse each other
# --------------------------------------------------------------------------- #
@pytest.fixture
def heads():
    return (
        ResponseLSTM(LSTMConfig(input_size=6, task=CLASSIFICATION)),
        ResponseLSTM(LSTMConfig(input_size=6, task=REGRESSION)),
    )


def test_a_regression_model_refuses_probability_methods(heads):
    _clf, reg = heads
    x = torch.randn(2, 4, 6)
    for method in (reg.predict_proba, reg.predict_tri):
        with pytest.raises(ValueError, match="classification"):
            method(x)


def test_a_classification_model_refuses_value_methods(heads):
    clf, _reg = heads
    x = torch.randn(2, 4, 6)
    for method in (clf.predict_value, clf.predict_value_sequence):
        with pytest.raises(ValueError, match="regression"):
            method(x)


def test_regression_output_is_unbounded(heads):
    """The head is linear: a BDI-II change of 30 points must be representable.

    A sigmoid would cap it at 1.0, which is why `predict_value` bypasses one.
    """
    _clf, reg = heads
    with torch.no_grad():
        reg.head.bias.fill_(30.0)
        reg.head.weight.zero_()
    assert reg.predict_value(torch.randn(3, 4, 6)).mean().item() == pytest.approx(30.0)


def test_an_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="task"):
        LSTMConfig(input_size=4, task="ranking")


def test_task_survives_a_checkpoint_round_trip(tmp_path):
    from src.models.train import load_model, save_model

    path = tmp_path / "reg.pt"
    save_model(ResponseLSTM(LSTMConfig(input_size=6, task=REGRESSION)), path)
    assert load_model(path).cfg.task == REGRESSION


def test_old_checkpoints_still_load_as_classification():
    """`task` is defaulted precisely so pre-regression checkpoints keep working."""
    assert LSTMConfig(input_size=8).task == CLASSIFICATION


# --------------------------------------------------------------------------- #
# 2. Targets and protocol filtering
# --------------------------------------------------------------------------- #
class _FakeDataset:
    """The narrow slice of `LoadedDataset` that `build_features` reads."""

    def __init__(self, n=10, n_epochs=3, window=8):
        import pandas as pd

        rng = np.random.default_rng(0)
        self.signals = rng.standard_normal((n, n_epochs, window)).astype(np.float32)
        self.signals_mc = None
        self.ecg = None
        self.channels = None
        self.fs = 250.0
        self.window = window
        self.labels = np.array([i % 2 for i in range(n)])
        self.metadata = pd.DataFrame({
            "patient_id": [f"P{i}" for i in range(n)],
            "protocol": [1 if i < 4 else 2 for i in range(n)],
            "age": np.linspace(30, 60, n),
            "gender": [i % 2 for i in range(n)],
            "bdi_pre": np.linspace(20, 45, n),
            "bdi_post": np.linspace(10, 20, n),
            "pct_reduction": np.linspace(0.1, 0.9, n),
        })


def test_delta_bdi_is_derived_when_the_column_is_absent():
    """The database round-trip keeps bdi_pre/bdi_post but not their difference."""
    ds = _FakeDataset()
    assert "delta_bdi" not in ds.metadata
    expected = ds.metadata.bdi_pre.to_numpy() - ds.metadata.bdi_post.to_numpy()
    np.testing.assert_allclose(target_values(ds, "delta_bdi"), expected)


def test_unknown_target_is_rejected():
    with pytest.raises(ValueError, match="unknown target"):
        target_values(_FakeDataset(), "vibes")


@pytest.mark.parametrize("protocol, expected", [(None, 10), (1, 4), (2, 6)])
def test_protocol_filter_selects_the_right_arm(protocol, expected):
    ds = _FakeDataset()
    x, y, groups, _ = build_features(
        ds, modalities=("rtms",), target="delta_bdi", protocol=protocol
    )
    assert x.shape[0] == y.shape[0] == len(groups) == expected


def test_protocol_filter_keeps_features_aligned_with_their_patients():
    """The mask is applied to x, y and groups together, or rows would shear apart."""
    ds = _FakeDataset()
    x_all, y_all, g_all, _ = build_features(ds, ("rtms",), target="delta_bdi")
    x_p1, y_p1, g_p1, _ = build_features(ds, ("rtms",), target="delta_bdi", protocol=1)

    mask = protocol_mask(ds, 1)
    np.testing.assert_allclose(x_p1, x_all[mask])
    np.testing.assert_allclose(y_p1, y_all[mask])
    np.testing.assert_array_equal(g_p1, g_all[mask])


def test_an_empty_protocol_is_refused_rather_than_returning_nothing():
    with pytest.raises(ValueError, match="protocol"):
        build_features(_FakeDataset(), ("rtms",), protocol=7)


# --------------------------------------------------------------------------- #
# 3. Cross-validation reports the right metric family
# --------------------------------------------------------------------------- #
def test_regression_cv_reports_r_not_auc():
    """The key sets are disjoint so a caller cannot average an AUC that isn't there."""
    rng = np.random.default_rng(0)
    n, seq, feats = 30, 3, 5
    x = rng.standard_normal((n, seq, feats)).astype(np.float32)
    y = (2.0 * x[:, :, 0].mean(axis=1)).astype(np.float32)
    cv = cross_validate(
        x, y, np.arange(n),
        lstm_cfg=LSTMConfig(input_size=feats, task=REGRESSION),
        train_cfg=TrainConfig(epochs=3), n_splits=3,
    )
    summary = cv.summary()
    assert cv.is_regression
    assert "r_mean" in summary and "auc_mean" not in summary


def test_repeats_multiply_the_folds_but_not_the_patients():
    """Repeated CV must not inflate n — that would inflate every p-value with it."""
    rng = np.random.default_rng(1)
    n, seq, feats = 24, 3, 4
    x = rng.standard_normal((n, seq, feats)).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    cv = cross_validate(
        x, y, np.arange(n),
        lstm_cfg=LSTMConfig(input_size=feats, task=REGRESSION),
        train_cfg=TrainConfig(epochs=2), n_splits=3, repeats=4,
    )
    assert len(cv.folds) == 12
    y_true, y_pred = cv.out_of_fold(y)
    assert len(y_true) == len(y_pred) == n


def test_repeats_must_be_positive():
    x = np.zeros((6, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="repeats"):
        cross_validate(x, np.zeros(6, dtype=np.float32), np.arange(6), repeats=0)


# --------------------------------------------------------------------------- #
# 4. The bar the EEG has to clear
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("protocol, floor", [(1, 0.40), (2, 0.25)])
def test_baseline_severity_alone_predicts_the_target(protocol, floor):
    """`bdi_pre` alone correlates strongly with `delta_bdi` — that is the bar.

    Protocol 1 measures r = 0.500, *above* the reference study's headline r = 0.401
    from EEG. This is mathematical coupling, not a finding: you cannot recover 40
    points from a BDI of 20. Any model reporting an r on this target has to be read
    against it, so it is pinned here rather than left to memory.

    Skipped when the real cohort is not seeded on this machine.
    """
    from pathlib import Path

    if not Path("recherche_tdbrain.sqlite3").exists():
        pytest.skip("real cohort not seeded on this machine")

    from src.data.tdbrain_seeder import dataset_from_repository
    from src.db import Repository

    ds = dataset_from_repository(Repository(db_url="sqlite:///recherche_tdbrain.sqlite3"))
    mask = protocol_mask(ds, protocol)
    delta = target_values(ds, "delta_bdi")[mask]
    bdi_pre = ds.metadata["bdi_pre"].to_numpy(float)[mask]
    assert pearson_r(delta, bdi_pre) > floor


def test_pct_reduction_is_the_less_confounded_target():
    """Dividing by baseline removes most of the coupling — the reason it is reported."""
    from pathlib import Path

    if not Path("recherche_tdbrain.sqlite3").exists():
        pytest.skip("real cohort not seeded on this machine")

    from src.data.tdbrain_seeder import dataset_from_repository
    from src.db import Repository

    ds = dataset_from_repository(Repository(db_url="sqlite:///recherche_tdbrain.sqlite3"))
    bdi_pre = ds.metadata["bdi_pre"].to_numpy(float)
    coupling_delta = abs(pearson_r(target_values(ds, "delta_bdi"), bdi_pre))
    coupling_pct = abs(pearson_r(target_values(ds, "pct_reduction"), bdi_pre))
    assert coupling_pct < coupling_delta


# --------------------------------------------------------------------------- #
# 5. The app serves the checkpoint the sidebar claims
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", [1, 2])
def test_model_choice_follows_the_selected_protocol(arm):
    """Selecting an arm must select that arm's checkpoint, not the pooled one.

    This failed silently once: `available_models`' `protocol` argument defaulted
    to `None`, which is also the legitimate value meaning "both arms pooled". So
    every page — all of which reach it through `model_choice()` without passing
    the argument — got the pooled checkpoint while the sidebar read "Protocole 1"
    and the patient list was correctly filtered to 44 protocol-1 patients. Nothing
    on screen looked wrong.
    """
    import streamlit as st

    from src.app.utils import SOURCES, DataSource, available_models, model_choice

    cfg = SOURCES[DataSource.TDBRAIN]
    st.session_state["data_source"] = DataSource.TDBRAIN.value
    st.session_state[f"protocole:{cfg.label}"] = arm
    st.session_state[f"model_choice:{cfg.label}"] = None

    assert all(m.protocol == arm for m in available_models(DataSource.TDBRAIN))
    assert model_choice(DataSource.TDBRAIN).protocol == arm


def test_the_sequential_cohort_keeps_its_classification_model():
    """The multi-session track is untouched by the article alignment.

    It is the only cohort where the LSTM accumulates evidence across real
    treatment sessions, so it stays binary and stays protocol-free — the
    per-arm regression axis simply does not apply to it.
    """
    import streamlit as st

    from src.app.utils import DataSource, current_protocol, model_choice

    st.session_state["data_source"] = DataSource.SIMULE_SEQ.value
    assert current_protocol(DataSource.SIMULE_SEQ) is None
    assert not model_choice(DataSource.SIMULE_SEQ).is_regression


def test_each_arm_offers_eeg_only_clinical_and_multimodal():
    """Three feature sets, and the first two are the comparison that matters.

    The article's model sees EEG *only*. The multimodal variant takes baseline
    BDI-II as an input, so its r cannot separate "the EEG predicted this" from
    "the intake form did" — which is precisely what EEG-only vs clinical-only
    answers.
    """
    import streamlit as st

    from src.app.utils import SOURCES, DataSource, available_models

    cfg = SOURCES[DataSource.TDBRAIN]
    st.session_state["data_source"] = DataSource.TDBRAIN.value
    for arm in (1, 2):
        st.session_state[f"protocole:{cfg.label}"] = arm
        options = available_models(DataSource.TDBRAIN)
        assert [m.label for m in options] == [
            "EEG seul (130)", "Clinique seul (4)", "Multimodal (139)",
        ]
        eeg_only = options[0]
        assert eeg_only.modalities == ("eeg",)
        assert "rtms" not in eeg_only.modalities, (
            "the article's model receives no clinical variables"
        )


# --------------------------------------------------------------------------- #
# 6. The leak that made a null result look like a discovery
# --------------------------------------------------------------------------- #
def test_early_stopping_never_watches_the_outer_fold():
    """The scored fold must not be the fold that chose the weights.

    `cross_validate` used to hand the outer held-out fold straight to
    `train_one_fold` as its early-stopping set, so each fold's checkpoint was
    selected on the very patients it was then scored on. On this cohort's
    near-constant regressor that tuned the emitted constant toward each held-out
    fold's own mean, and the pooled out-of-fold correlation reached r = 0.61 on
    **shuffled** labels — briefly reading as though the model had beaten the
    reference study.
    """
    from src.models.train import _inner_split

    groups = np.array([f"P{i}" for i in range(20)])
    tr_idx = np.arange(15)
    fit_idx, stop_idx = _inner_split(tr_idx, groups, seed=0)

    assert set(fit_idx) | set(stop_idx) == set(tr_idx)      # drawn from training only
    assert not set(fit_idx) & set(stop_idx)                 # and disjoint
    assert len(stop_idx) >= 1
    outer = set(range(15, 20))
    assert not (set(fit_idx) | set(stop_idx)) & outer       # outer fold untouched


def test_inner_split_keeps_a_patient_whole():
    """A patient with several rows lands entirely on one side, like the outer split."""
    from src.models.train import _inner_split

    groups = np.repeat([f"P{i}" for i in range(10)], 3)
    fit_idx, stop_idx = _inner_split(np.arange(len(groups)), groups, seed=1)
    assert not set(groups[fit_idx]) & set(groups[stop_idx])


def test_shuffled_labels_do_not_produce_a_correlation():
    """The end-to-end leak check: retrain on permuted targets, expect nothing.

    A permutation test applied *after* prediction only checks the statistic. This
    retrains the whole pipeline on shuffled labels, which is what actually
    catches a fold-level leak. Deliberately generous bounds — with a
    near-constant predictor the pooled r on a small cohort is a noisy statistic —
    but the leaked version cleared 0.6 repeatedly, so the bound still bites.
    """
    from src.models.metrics import pearson_r

    rng = np.random.default_rng(0)
    n, seq, feats = 40, 4, 6
    x = rng.standard_normal((n, seq, feats)).astype(np.float32)
    groups = np.arange(n)

    for i in range(2):
        y_shuffled = rng.permutation(
            rng.standard_normal(n).astype(np.float32) * 12.0
        )
        cv = cross_validate(
            x, y_shuffled, groups,
            lstm_cfg=LSTMConfig(input_size=feats, task=REGRESSION),
            train_cfg=TrainConfig(epochs=10, seed=1), n_splits=5, repeats=3,
        )
        y_true, y_pred = cv.out_of_fold(y_shuffled)
        r = pearson_r(y_true, y_pred)
        assert not np.isfinite(r) or abs(r) < 0.55, (
            f"run {i}: r={r:+.3f} on shuffled labels suggests information is "
            f"reaching the scored fold"
        )
