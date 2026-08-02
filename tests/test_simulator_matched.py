"""Tests for the TDBRAIN-matched simulator.

Two families of guarantee here, and the second is the one that matters:

1. **Contract** — the generated cohort must be interchangeable with
   ``load_tdbrain``'s output, or the four-model comparison is not like-for-like.
2. **Calibration** — the cohort must reproduce the *real* cohort's behaviour: age
   separates responders, baseline BDI-II and protocol do not, and the
   neurophysiological blocks carry no label information until ``effect_size`` is
   dialled up. A simulator that leaked signal through EEG at ``effect_size=0``
   would invalidate its whole purpose as a positive control.

The cohort is expensive to generate (26 channels x 8 epochs x 2000 samples), so
the fixtures use a reduced patient count and are module-scoped.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import LoadedDataset
from src.data.simulator_matched import (
    TARGET_BAND_POWER,
    TARGET_HR_MEAN,
    MatchedSimConfig,
    simulate_matched,
)
from src.data.tdbrain import TDBRAIN_CHANNELS_26, tdbrain_features
from src.preprocessing.features import BANDS, hrv_features
from src.preprocessing.pipeline import HRV_FEATURE_NAMES

SMALL = dict(n_patients=40, n_epochs=4, window=512, seed=3)


def _auc(x, y, splits=4):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return cross_val_score(
        model, x, y, scoring="roc_auc",
        cv=StratifiedKFold(splits, shuffle=True, random_state=0),
    ).mean()


@pytest.fixture(scope="module")
def null_ds():
    """effect_size = 0 — reproduces the real cohort's null result."""
    return simulate_matched(MatchedSimConfig(effect_size=0.0, **SMALL))


@pytest.fixture(scope="module")
def signal_ds():
    """effect_size > 0 — a known effect the pipeline should be able to find."""
    return simulate_matched(MatchedSimConfig(effect_size=1.0, **SMALL))


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


def test_returns_a_loaded_dataset_shaped_like_tdbrain(null_ds):
    assert isinstance(null_ds, LoadedDataset)
    n = SMALL["n_patients"]
    assert null_ds.signals_mc.shape == (n, 4, len(TDBRAIN_CHANNELS_26), 512)
    assert null_ds.signals.shape == (n, 4, 512)
    assert null_ds.ecg.shape == (n, 4, 64)
    assert null_ds.channels == list(TDBRAIN_CHANNELS_26)
    assert null_ds.erp is None
    assert np.isfinite(null_ds.signals_mc).all()


def test_metadata_carries_the_clinical_model_columns(null_ds):
    required = {
        "patient_id", "protocol", "bdi_pre", "bdi_post", "delta_bdi",
        "pct_reduction", "responder", "age", "gender",
    }
    assert required <= set(null_ds.metadata.columns)
    assert null_ds.metadata["patient_id"].nunique() == SMALL["n_patients"]


def test_label_always_agrees_with_the_bdi_reduction(null_ds):
    """A stored label that disagrees with the scores would corrupt every metric."""
    md = null_ds.metadata
    derived = (md["pct_reduction"] >= 0.5).astype(int)
    assert (derived == md["responder"]).all()
    assert (null_ds.labels == md["responder"].to_numpy()).all()


def test_feeds_the_tdbrain_feature_pipeline_unchanged(null_ds):
    x, y, groups, names = tdbrain_features(
        null_ds, per_patient_zscore=False, modalities=("eeg", "ecg")
    )
    assert x.shape == (SMALL["n_patients"], 4, len(TDBRAIN_CHANNELS_26) * 5 + 5)
    assert names[-5:] == tuple(HRV_FEATURE_NAMES)
    assert len(set(groups)) == SMALL["n_patients"]
    assert np.isfinite(x).all()


def test_is_deterministic_for_a_given_seed():
    a = simulate_matched(MatchedSimConfig(n_patients=6, n_epochs=2, window=256, seed=11))
    b = simulate_matched(MatchedSimConfig(n_patients=6, n_epochs=2, window=256, seed=11))
    np.testing.assert_array_equal(a.signals_mc, b.signals_mc)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_hrv_is_constant_across_epochs(null_ds):
    """Patient-level trait, exactly as the real loader stores it."""
    for p in range(null_ds.ecg.shape[0]):
        np.testing.assert_allclose(null_ds.ecg[p, 0], null_ds.ecg[p, 1])


# --------------------------------------------------------------------------- #
# Calibration against the real cohort
# --------------------------------------------------------------------------- #


def test_responder_rate_matches_the_real_cohort(null_ds):
    assert null_ds.labels.mean() == pytest.approx(0.634, abs=0.06)


def test_band_powers_match_the_real_distributions(null_ds):
    x, _, _, _ = tdbrain_features(null_ds, per_patient_zscore=False, modalities=("eeg",))
    for i, band in enumerate(BANDS):
        block = x[:, :, i::5]
        target_mu = TARGET_BAND_POWER[band][0]
        assert block.mean() == pytest.approx(target_mu, abs=0.06), band


def test_heart_rate_matches_the_real_distribution(null_ds):
    hr = np.array([
        hrv_features(null_ds.ecg[p, 0])["hrv_hr_mean"]
        for p in range(null_ds.ecg.shape[0])
    ])
    assert hr.mean() == pytest.approx(TARGET_HR_MEAN[0], abs=4.0)
    assert 45.0 < hr.min() and hr.max() < 115.0


def test_age_separates_responders_as_it_does_in_reality(null_ds):
    """Responders are younger in the real cohort (42.5 vs 46.8); reproduce that."""
    md = null_ds.metadata
    assert md[md.responder == 1].age.mean() < md[md.responder == 0].age.mean()


def test_baseline_bdi_does_not_separate_responders(null_ds):
    """Real cohort: 30.8 vs 32.1 — no usable baseline-severity signal."""
    md = null_ds.metadata
    gap = abs(md[md.responder == 1].bdi_pre.mean() - md[md.responder == 0].bdi_pre.mean())
    assert gap < 6.0


def test_protocol_is_independent_of_outcome(null_ds):
    """Real cohort: chi2 p = 0.885. A simulator that coupled them would invent
    a dose-response relationship the data does not support."""
    md = null_ds.metadata
    rate1 = md[md.protocol == 1].responder.mean()
    rate2 = md[md.protocol == 2].responder.mean()
    assert abs(rate1 - rate2) < 0.30


# --------------------------------------------------------------------------- #
# The positive control
# --------------------------------------------------------------------------- #


def test_null_cohort_gives_the_pipeline_no_eeg_signal(null_ds):
    """At effect_size = 0 the EEG must be uninformative, as on real data."""
    x, _, _, _ = tdbrain_features(null_ds, per_patient_zscore=False, modalities=("eeg",))
    assert _auc(x.mean(axis=1), null_ds.labels.astype(int)) < 0.70


def test_effect_cohort_is_detectable(signal_ds):
    """At effect_size = 1 the same pipeline must find it — otherwise a null on
    real data could not be distinguished from a broken pipeline."""
    x, _, _, _ = tdbrain_features(signal_ds, per_patient_zscore=False, modalities=("eeg",))
    assert _auc(x.mean(axis=1), signal_ds.labels.astype(int)) > 0.70


def test_effect_size_is_the_only_thing_that_changes_eeg_separability(null_ds, signal_ds):
    y_null = null_ds.labels.astype(int)
    y_sig = signal_ds.labels.astype(int)
    xn, _, _, _ = tdbrain_features(null_ds, per_patient_zscore=False, modalities=("eeg",))
    xs, _, _, _ = tdbrain_features(signal_ds, per_patient_zscore=False, modalities=("eeg",))
    assert _auc(xs.mean(axis=1), y_sig) > _auc(xn.mean(axis=1), y_null) + 0.10
