"""Tests for the multimodal (EEG + ERP + ECG) extension, TRI trajectory, and ablation.

These mirror the NPDT design in RES0_AR1: cognitive (ERP) + autonomic (ECG)
channels feeding a physics-agnostic response model, a per-session Therapeutic
Response Index, and a modality ablation to justify each channel.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.loader import load
from src.data.simulator import SimConfig, save, simulate
from src.models.ablation import DEFAULT_MODALITY_SETS, run_ablation
from src.models.lstm import LSTMConfig, ResponseLSTM
from src.models.train import TrainConfig
from src.preprocessing.features import erp_features, hrv_features
from src.preprocessing.pipeline import (
    ERP_FEATURE_NAMES,
    FEATURE_NAMES,
    HRV_FEATURE_NAMES,
    MultimodalConfig,
    preprocess_multimodal,
)

FS = 256.0


# --- simulator --------------------------------------------------------------

def test_simulate_multimodal_shapes():
    ds = simulate(SimConfig(n_patients=8, n_sessions=6, window=128, n_rr=64, seed=0))
    assert ds.erp is not None and ds.ecg is not None
    assert ds.erp.shape == (8, 6, 128)
    assert ds.ecg.shape == (8, 6, 64)
    assert ds.erp.dtype == np.float32 and ds.ecg.dtype == np.float32
    assert {"erp_n100_uv", "hrv_sdnn_ms"} <= set(ds.metadata.columns)


def test_multimodal_off_leaves_eeg_identical():
    """Disabling multimodal must not perturb the EEG stream (independent RNG)."""
    a = simulate(SimConfig(n_patients=6, n_sessions=5, window=64, seed=3, multimodal=True))
    b = simulate(SimConfig(n_patients=6, n_sessions=5, window=64, seed=3, multimodal=False))
    np.testing.assert_array_equal(a.signals, b.signals)
    assert b.erp is None and b.ecg is None


def test_responders_have_higher_late_hrv_and_erp():
    """Responder recovery must be learnable in the ERP + ECG channels, not just EEG."""
    ds = simulate(SimConfig(n_patients=120, n_sessions=10, seed=5))

    sdnn = ds.ecg[:, -1, :].std(axis=-1)
    assert sdnn[ds.labels == 1].mean() > sdnn[ds.labels == 0].mean()

    n100 = -ds.erp[:, -1, :].min(axis=-1)  # deeper trough → larger N100
    assert n100[ds.labels == 1].mean() > n100[ds.labels == 0].mean()


def test_multimodal_save_load_roundtrip(tmp_path):
    ds = simulate(SimConfig(n_patients=5, n_sessions=3, window=64, seed=7))
    save(ds, tmp_path)
    loaded = load(tmp_path)
    np.testing.assert_array_equal(loaded.erp, ds.erp)
    np.testing.assert_array_equal(loaded.ecg, ds.ecg)


# --- features ---------------------------------------------------------------

def test_erp_features_locate_n100_and_p300():
    t = np.arange(128) / FS
    wave = (
        -1.0 * np.exp(-0.5 * ((t - 0.10) / 0.02) ** 2)
        + 0.8 * np.exp(-0.5 * ((t - 0.30) / 0.05) ** 2)
    )
    feats = erp_features(wave, FS)
    assert set(feats) == set(ERP_FEATURE_NAMES)
    assert abs(feats["erp_n100_lat"] - 0.10) < 0.02
    assert abs(feats["erp_p300_lat"] - 0.30) < 0.03
    assert feats["erp_n100_amp"] > 0.5


def test_hrv_features_increase_with_variability():
    rng = np.random.default_rng(0)
    low = 0.8 + 0.01 * rng.standard_normal(64)
    high = 0.8 + 0.06 * rng.standard_normal(64)
    f_low, f_high = hrv_features(low), hrv_features(high)
    assert set(f_low) == set(HRV_FEATURE_NAMES)
    assert f_high["hrv_sdnn"] > f_low["hrv_sdnn"]
    assert f_high["hrv_rmssd"] > f_low["hrv_rmssd"]
    assert 40 < f_low["hrv_hr_mean"] < 100


# --- multimodal pipeline ----------------------------------------------------

def test_preprocess_multimodal_feature_count_and_names():
    ds = simulate(SimConfig(n_patients=6, n_sessions=4, window=128, seed=1))
    out = preprocess_multimodal(ds.signals, ds.erp, ds.ecg, MultimodalConfig(fs=FS))
    expected = len(FEATURE_NAMES) + len(ERP_FEATURE_NAMES) + len(HRV_FEATURE_NAMES)
    assert out.x.shape == (6, 4, expected)
    assert out.feature_names is not None and len(out.feature_names) == expected
    assert np.allclose(out.x.mean(axis=1), 0.0, atol=1e-4)  # per-patient z-score


def test_preprocess_multimodal_modality_subset():
    ds = simulate(SimConfig(n_patients=4, n_sessions=3, window=128, seed=2))
    eeg_only = preprocess_multimodal(ds.signals, ds.erp, ds.ecg,
                                     MultimodalConfig(fs=FS, modalities=("eeg",)))
    eeg_ecg = preprocess_multimodal(ds.signals, ds.erp, ds.ecg,
                                    MultimodalConfig(fs=FS, modalities=("eeg", "ecg")))
    assert eeg_only.x.shape[-1] == len(FEATURE_NAMES)
    assert eeg_ecg.x.shape[-1] == len(FEATURE_NAMES) + len(HRV_FEATURE_NAMES)


def test_preprocess_multimodal_missing_array_raises():
    ds = simulate(SimConfig(n_patients=3, n_sessions=2, window=64, seed=0, multimodal=False))
    with pytest.raises(ValueError):
        preprocess_multimodal(ds.signals, None, None, MultimodalConfig(modalities=("eeg", "erp")))


# --- TRI trajectory ---------------------------------------------------------

def test_predict_tri_shape_and_range():
    model = ResponseLSTM(LSTMConfig(input_size=8))
    x = torch.randn(4, 10, 8)
    tri = model.predict_tri(x)
    assert tri.shape == (4, 10)
    assert torch.all((tri >= 0) & (tri <= 1))
    # the last session's TRI must equal the standard responder probability
    torch.testing.assert_close(tri[:, -1], model.predict_proba(x))


# --- ablation ---------------------------------------------------------------

def test_run_ablation_returns_row_per_modality_set():
    ds = simulate(SimConfig(n_patients=40, n_sessions=10, window=128, seed=0))
    groups = np.arange(ds.signals.shape[0])
    rows = run_ablation(
        ds.signals, ds.labels, groups, erp=ds.erp, ecg=ds.ecg, fs=FS,
        train_cfg=TrainConfig(epochs=4, batch_size=8, lr=5e-3, early_stopping_patience=3, seed=0),
        n_splits=3,
    )
    assert len(rows) == len(DEFAULT_MODALITY_SETS)
    for r in rows:
        assert r.n_features > 0
        assert 0.0 <= r.accuracy_mean <= 1.0
        assert np.isfinite(r.auc_mean)
    assert rows[0].n_features < rows[-1].n_features  # EEG-only ⊂ full multimodal
