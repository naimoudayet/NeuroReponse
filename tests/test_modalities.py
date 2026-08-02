"""Tests for the unified modality feature layer.

This layer is what makes the four-model comparison honest: the real and the
simulated cohort must be turned into model inputs by the *same* code. The tests
therefore run the same assertions against both sources wherever possible.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.data.modalities import (
    MODALITY_ORDER,
    RTMS_FEATURE_NAMES,
    ModalityError,
    build_features,
    feature_dimension,
)
from src.data.simulator_matched import MatchedSimConfig, simulate_matched
from src.data.tdbrain import (
    TDBRAIN_CHANNELS_26,
    TDBRAINConfig,
    load_tdbrain,
    make_synthetic_tdbrain,
)

N_EEG = len(TDBRAIN_CHANNELS_26) * 5
N_ECG = 5
N_RTMS = len(RTMS_FEATURE_NAMES)


@pytest.fixture(scope="module")
def sim_ds():
    return simulate_matched(
        MatchedSimConfig(n_patients=16, n_epochs=4, window=512, seed=17)
    )


@pytest.fixture(scope="module")
def real_ds(tmp_path_factory):
    """A cohort through the *real* loader, so both paths are exercised."""
    root = make_synthetic_tdbrain(
        tmp_path_factory.mktemp("td") / "t", n_patients=8, seed=5,
        with_ecg=True, duration_seconds=20.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_tdbrain(
            TDBRAINConfig(root=root, n_epochs=4, epoch_seconds=1.0, target_fs=250.0)
        )


# --------------------------------------------------------------------------- #
# Block shapes and naming
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mods,width", [
    (("rtms",), N_RTMS),
    (("eeg",), N_EEG),
    (("ecg",), N_ECG),
    (("rtms", "eeg"), N_RTMS + N_EEG),
    (("eeg", "ecg"), N_EEG + N_ECG),
    (("rtms", "eeg", "ecg"), N_RTMS + N_EEG + N_ECG),
])
def test_block_widths_on_simulated(sim_ds, mods, width):
    x, y, groups, names = build_features(sim_ds, modalities=mods)
    assert x.shape == (16, 4, width)
    assert len(names) == width
    assert feature_dimension(sim_ds, mods) == width
    assert len(set(groups)) == 16


def test_same_widths_on_the_real_loader(real_ds):
    x, _, _, names = build_features(real_ds, modalities=("rtms", "eeg", "ecg"))
    assert x.shape[-1] == N_RTMS + N_EEG + N_ECG
    assert names[:N_RTMS] == RTMS_FEATURE_NAMES
    assert np.isfinite(x).all()


def test_block_order_is_canonical_not_argument_order(sim_ds):
    """A permuted feature vector is silent and fatal; order must not depend on
    how the caller spells the modality tuple."""
    a, _, _, na = build_features(sim_ds, modalities=("ecg", "rtms", "eeg"))
    b, _, _, nb = build_features(sim_ds, modalities=("rtms", "eeg", "ecg"))
    np.testing.assert_array_equal(a, b)
    assert na == nb
    assert na[0] == "rtms_protocol"
    assert na[-1] == "hrv_lf_hf"


def test_modality_order_constant_matches_reality():
    assert MODALITY_ORDER == ("rtms", "eeg", "ecg")


# --------------------------------------------------------------------------- #
# The normalisation trap — now threatening two patient-level blocks
# --------------------------------------------------------------------------- #


def test_zscoring_does_not_blank_the_clinical_or_hrv_blocks(sim_ds):
    """Both the rtms and ecg blocks are constant across epochs. Routing either
    through zscore_epochs would divide by a zero std and zero the block out,
    deleting a modality while every shape assertion still passes."""
    x, _, _, _ = build_features(
        sim_ds, modalities=("rtms", "eeg", "ecg"), per_patient_zscore=True
    )
    rtms = x[..., :N_RTMS]
    hrv = x[..., N_RTMS + N_EEG:]

    assert not np.allclose(rtms, 0.0)
    assert not np.allclose(hrv, 0.0)
    # Still in physical units, not standardised.
    assert rtms[..., 1].mean() > 20.0          # age, in years
    assert hrv[..., 0].mean() > 20.0           # mean heart rate, bpm


def test_patient_level_blocks_are_constant_across_epochs(sim_ds):
    x, _, _, _ = build_features(sim_ds, modalities=("rtms", "eeg", "ecg"))
    for p in range(x.shape[0]):
        np.testing.assert_allclose(x[p, 0, :N_RTMS], x[p, 1, :N_RTMS])
        np.testing.assert_allclose(x[p, 0, N_RTMS + N_EEG:], x[p, 1, N_RTMS + N_EEG:])


def test_eeg_block_is_z_scored_when_requested(sim_ds):
    raw, _, _, _ = build_features(sim_ds, modalities=("eeg",), per_patient_zscore=False)
    zed, _, _, _ = build_features(sim_ds, modalities=("eeg",), per_patient_zscore=True)
    assert not np.allclose(raw, zed)
    assert abs(float(zed.mean())) < 1e-4        # centred within patient


def test_adding_a_block_leaves_the_others_untouched(sim_ds):
    only_eeg, _, _, _ = build_features(sim_ds, modalities=("eeg",))
    with_all, _, _, _ = build_features(sim_ds, modalities=("rtms", "eeg", "ecg"))
    np.testing.assert_allclose(with_all[..., N_RTMS:N_RTMS + N_EEG], only_eeg)


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


def test_unknown_modality_is_rejected(sim_ds):
    with pytest.raises(ModalityError, match="unknown modalities"):
        build_features(sim_ds, modalities=("eeg", "fnirs"))


def test_empty_modality_tuple_is_rejected(sim_ds):
    with pytest.raises(ModalityError, match="at least one"):
        build_features(sim_ds, modalities=())


def test_missing_clinical_columns_named_in_the_error(sim_ds):
    ds = simulate_matched(MatchedSimConfig(n_patients=6, n_epochs=2, window=256))
    ds.metadata = ds.metadata.drop(columns=["age", "bdi_pre"])
    with pytest.raises(ModalityError, match="age"):
        build_features(ds, modalities=("rtms",))


def test_ecg_requested_without_a_tachogram(sim_ds):
    ds = simulate_matched(MatchedSimConfig(n_patients=6, n_epochs=2, window=256))
    ds.ecg = None
    with pytest.raises(ModalityError, match="tachogram"):
        build_features(ds, modalities=("ecg",))


def test_missing_clinical_values_are_imputed_with_a_warning():
    ds = simulate_matched(MatchedSimConfig(n_patients=8, n_epochs=2, window=256))
    ds.metadata.loc[0, "age"] = np.nan
    with pytest.warns(UserWarning, match="imputed"):
        x, _, _, _ = build_features(ds, modalities=("rtms",))
    assert np.isfinite(x).all()
