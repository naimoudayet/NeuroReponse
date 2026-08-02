"""Tests for the real-data TDBRAIN loader.

These exercise the parser + feature path against a synthetic TDBRAIN-format tree
(no patient data ever touches the repo — see docs/tdbrain.md). They enforce the
same guarantees the simulated pipeline has: correct LoadedDataset contract,
patient-wise grouping, and no cross-patient leakage in normalization.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.data.loader import LoadedDataset
from src.data.tdbrain import (
    TDBRAIN_CHANNELS_26,
    TDBRAINConfig,
    load_tdbrain,
    make_synthetic_tdbrain,
    tdbrain_features,
)


@pytest.fixture
def synth_root(tmp_path):
    return make_synthetic_tdbrain(tmp_path / "tdbrain", n_patients=12, seed=1)


@pytest.fixture
def synth_bdf_root(tmp_path):
    # Real TDBRAIN ships BioSemi BDF, not CSV. Writing BDF needs mne + edfio;
    # skip cleanly where they aren't installed so the CSV suite still runs.
    pytest.importorskip("mne")
    pytest.importorskip("edfio")
    return make_synthetic_tdbrain(tmp_path / "tdbrain_bdf", n_patients=8, seed=2, fmt="bdf")


def _cfg(root, **kw):
    base = dict(root=root, n_epochs=4, epoch_seconds=1.0, target_fs=250.0)
    base.update(kw)
    return TDBRAINConfig(**base)


def test_load_returns_loaded_dataset_contract(synth_root):
    ds = load_tdbrain(_cfg(synth_root))
    assert isinstance(ds, LoadedDataset)
    n_patients = ds.signals.shape[0]
    assert ds.signals.shape == (n_patients, 4, 250)                 # (patients, epochs, window)
    assert ds.signals_mc.shape == (n_patients, 4, len(TDBRAIN_CHANNELS_26), 250)
    assert ds.labels.shape == (n_patients,)
    assert set(np.unique(ds.labels)).issubset({0, 1})
    assert ds.fs == 250.0 and ds.window == 250
    assert ds.erp is None and ds.ecg is None                       # EEG-only for real data
    assert ds.channels == list(TDBRAIN_CHANNELS_26)


def test_noise_rows_are_filtered(synth_root):
    # 12 eligible MDD/protocol-{1,2} patients; the protocol-3 and ADHD rows must drop.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = load_tdbrain(_cfg(synth_root))
    assert ds.signals.shape[0] == 12
    assert set(ds.metadata["protocol"].unique()) == {1, 2}
    assert "19000001" not in set(ds.metadata["patient_id"])        # protocol 3
    assert "19000002" not in set(ds.metadata["patient_id"])        # ADHD / no BDI


def test_responder_label_matches_50pct_bdi_rule(synth_root):
    ds = load_tdbrain(_cfg(synth_root))
    pct = ds.metadata["pct_reduction"].to_numpy()
    expected = (pct >= 0.5).astype(int)
    assert np.array_equal(ds.metadata["responder"].to_numpy(), expected)
    assert np.array_equal(ds.labels.astype(int), expected)
    # Synthetic responders get 75% reduction, non-responders 10%.
    assert ds.labels.sum() == 6


def test_protocol_filter_selects_single_protocol(synth_root):
    ds = load_tdbrain(_cfg(synth_root, protocols=(1,)))
    assert set(ds.metadata["protocol"].unique()) == {1}


def test_snapshot_mode_collapses_sequence(synth_root):
    ds = load_tdbrain(_cfg(synth_root, snapshot=True))
    assert ds.signals.shape[1] == 1                                # one window per patient
    assert ds.window == 4 * 250                                    # n_epochs * epoch samples
    x, _, _, _ = tdbrain_features(ds)
    assert x.shape[1] == 1


def test_eyes_open_and_closed_select_different_files(synth_root):
    eo = load_tdbrain(_cfg(synth_root, condition="EO"))
    ec = load_tdbrain(_cfg(synth_root, condition="EC"))
    # EO carries the discriminative alpha; the two conditions must not be identical.
    assert not np.allclose(eo.signals_mc, ec.signals_mc)


def test_features_shape_and_names(synth_root):
    ds = load_tdbrain(_cfg(synth_root))
    x, y, groups, names = tdbrain_features(ds)
    n_patients = ds.signals.shape[0]
    assert x.shape == (n_patients, 4, len(TDBRAIN_CHANNELS_26) * 5)
    assert len(names) == x.shape[-1]
    assert names[0] == "Fp1_power_delta"
    assert len(groups) == n_patients
    assert len(set(groups)) == n_patients                          # one group per patient


def test_no_cross_patient_leakage_in_normalization(synth_root):
    """Rescaling one patient's raw EEG must not change any other patient's features."""
    ds = load_tdbrain(_cfg(synth_root))
    x_ref, _, _, _ = tdbrain_features(ds)

    perturbed = LoadedDataset(
        signals=ds.signals.copy(),
        labels=ds.labels,
        fs=ds.fs,
        window=ds.window,
        metadata=ds.metadata,
        channels=ds.channels,
        signals_mc=ds.signals_mc.copy(),
    )
    perturbed.signals_mc[0] *= 5.0                                 # scale patient 0 only
    x_pert, _, _, _ = tdbrain_features(perturbed)

    assert not np.allclose(x_ref[0], x_pert[0])                    # patient 0 changed...
    assert np.allclose(x_ref[1:], x_pert[1:])                     # ...everyone else identical


def test_end_to_end_cross_validate_runs(synth_root):
    from src.models.train import TrainConfig, cross_validate

    ds = load_tdbrain(_cfg(synth_root))
    x, y, groups, _ = tdbrain_features(ds)
    result = cross_validate(
        x, y, groups,
        train_cfg=TrainConfig(epochs=2, batch_size=4),
        n_splits=3,
    )
    assert len(result.folds) == 3
    summary = result.summary()
    assert np.isfinite(summary["accuracy_mean"])                  # pipeline runs on real-shaped data


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tdbrain(TDBRAINConfig(root=tmp_path / "does-not-exist"))


# --------------------------------------------------------------------------- #
# Real-format (BDF) path — the actual TDBRAIN download is BDF, not CSV.
# --------------------------------------------------------------------------- #

def test_bdf_tree_loads_through_full_pipeline(synth_bdf_root):
    """The loader must read real-format .bdf recordings end-to-end, same contract."""
    # The fixture wrote BDF (real format), not CSV.
    assert list(synth_bdf_root.rglob("*.bdf"))
    assert not list(synth_bdf_root.rglob("*.csv"))

    ds = load_tdbrain(_cfg(synth_bdf_root))
    assert isinstance(ds, LoadedDataset)
    n_patients = ds.signals.shape[0]
    assert ds.signals_mc.shape == (n_patients, 4, len(TDBRAIN_CHANNELS_26), 250)
    assert ds.channels == list(TDBRAIN_CHANNELS_26)
    assert ds.erp is None and ds.ecg is None

    x, y, groups, _ = tdbrain_features(ds)
    assert x.shape == (n_patients, 4, len(TDBRAIN_CHANNELS_26) * 5)
    assert np.isfinite(x).all()
    assert set(np.unique(y)).issubset({0, 1})


def test_bdf_bandpass_config_is_applied(synth_bdf_root):
    """The notch/band-pass config must actually filter the BDF signal."""
    from src.data.tdbrain import SOURCE_FS, _read_condition_bdf
    from src.preprocessing.features import BANDS, basic_features

    rec = sorted(synth_bdf_root.rglob("*restEO*.bdf"))[0]
    unfiltered = _read_condition_bdf(rec, TDBRAIN_CHANNELS_26, notch_hz=None, bandpass_hz=None)
    filtered = _read_condition_bdf(rec, TDBRAIN_CHANNELS_26, notch_hz=None, bandpass_hz=(1.0, 45.0))

    assert unfiltered.shape == filtered.shape
    assert not np.allclose(unfiltered, filtered)            # filtering changed the signal

    def in_band_fraction(sig):
        feats = basic_features(sig, SOURCE_FS)
        return sum(feats[f"power_{b}"] for b in BANDS)

    # Removing broadband noise above 45 Hz raises the in-band power fraction.
    assert in_band_fraction(filtered[20]) >= in_band_fraction(unfiltered[20]) - 1e-6


# --------------------------------------------------------------------------- #
# Autonomic (ECG) modality — TDBRAIN ships an Erb's-point lead in every
# recording, so HRV is real data here, not a simulated stand-in.
# --------------------------------------------------------------------------- #


@pytest.fixture
def synth_ecg_root(tmp_path):
    return make_synthetic_tdbrain(
        tmp_path / "tdbrain_ecg", n_patients=8, seed=3, with_ecg=True,
        duration_seconds=30.0,
    )


def test_detect_rr_recovers_a_known_heart_rate():
    """R-peak detection must land on the true rate, T wave notwithstanding."""
    from src.data.tdbrain import SOURCE_FS, _synthetic_qrs, detect_rr_intervals

    rng = np.random.default_rng(0)
    ecg = _synthetic_qrs(int(60 * SOURCE_FS), SOURCE_FS, bpm=72.0, rng=rng)
    rr = detect_rr_intervals(ecg, SOURCE_FS)

    assert rr.size > 40
    assert 60.0 / np.mean(rr) == pytest.approx(72.0, abs=4.0)
    # A T wave counted as a beat would roughly double the rate; assert it did not.
    assert 60.0 / np.mean(rr) < 110.0


def test_detect_rr_rejects_flat_and_short_input():
    from src.data.tdbrain import detect_rr_intervals

    assert detect_rr_intervals(np.zeros(5000), 500.0).size == 0
    assert detect_rr_intervals(np.ones(50), 500.0).size == 0


def test_malik_filter_drops_doubled_intervals():
    from src.data.tdbrain import _malik_filter

    rr = np.array([0.85, 0.86, 1.70, 0.84, 0.85, 0.43, 0.86])   # one missed, one spurious
    kept = _malik_filter(rr)
    assert kept.size == 5
    assert kept.max() < 1.0 and kept.min() > 0.6


def test_loader_populates_the_tachogram(synth_ecg_root):
    ds = load_tdbrain(_cfg(synth_ecg_root))
    n_patients = ds.signals.shape[0]

    assert ds.ecg is not None
    assert ds.ecg.shape == (n_patients, 4, 64)
    assert np.isfinite(ds.ecg).all()
    # HRV is a patient-level trait: identical on every epoch, by construction.
    for p in range(n_patients):
        assert np.allclose(ds.ecg[p, 0], ds.ecg[p, 1])


def test_missing_ecg_yields_none_not_zeros(synth_root):
    """A cohort with no autonomic lead must report absence, not a block of zeros."""
    ds = load_tdbrain(_cfg(synth_root))
    assert ds.ecg is None
    with pytest.raises(ValueError, match="ecg"):
        tdbrain_features(ds, modalities=("eeg", "ecg"))


def test_multimodal_features_concatenate_eeg_and_hrv(synth_ecg_root):
    ds = load_tdbrain(_cfg(synth_ecg_root))
    n_eeg = len(TDBRAIN_CHANNELS_26) * 5

    x_eeg, _, _, names_eeg = tdbrain_features(ds, modalities=("eeg",))
    x_both, _, _, names_both = tdbrain_features(ds, modalities=("eeg", "ecg"))

    assert x_eeg.shape[-1] == n_eeg
    assert x_both.shape[-1] == n_eeg + 5
    assert names_both[:n_eeg] == names_eeg
    assert names_both[-5:] == (
        "hrv_hr_mean", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_lf_hf",
    )
    # The EEG block must be untouched by the presence of the autonomic block.
    np.testing.assert_allclose(x_both[..., :n_eeg], x_eeg)
    assert np.isfinite(x_both).all()


def test_zscoring_does_not_blank_the_hrv_block(synth_ecg_root):
    """The trap: HRV is constant across epochs, so z-scoring it would zero it out.

    Regression guard — if the HRV block is ever routed through zscore_epochs the
    modality silently disappears while every shape assertion still passes.
    """
    ds = load_tdbrain(_cfg(synth_ecg_root))
    n_eeg = len(TDBRAIN_CHANNELS_26) * 5
    x, _, _, _ = tdbrain_features(ds, per_patient_zscore=True, modalities=("eeg", "ecg"))

    hrv = x[..., n_eeg:]
    assert not np.allclose(hrv, 0.0)
    assert hrv[..., 0].min() > 20.0          # mean heart rate stays in physical units


def test_unknown_modality_is_rejected(synth_ecg_root):
    ds = load_tdbrain(_cfg(synth_ecg_root))
    with pytest.raises(ValueError, match="unknown modalities"):
        tdbrain_features(ds, modalities=("eeg", "fnirs"))
