from __future__ import annotations

import numpy as np
import pytest

from src.data.simulator import SimConfig, simulate
from src.domain import Preprocessing
from src.preprocessing.features import basic_features
from src.preprocessing.filters import bandpass
from src.preprocessing.pipeline import FEATURE_NAMES, PipelineConfig, preprocess
from src.preprocessing.windowing import sliding_windows


FS = 256.0


def _sine(freq: float, n: int = 1024, amp: float = 1.0, fs: float = FS) -> np.ndarray:
    t = np.arange(n) / fs
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def test_bandpass_preserves_in_band_sine():
    x = _sine(10.0)
    y = bandpass(x, 1.0, 40.0, FS)
    # In-band power should be preserved (~ within 5%).
    assert abs(y.std() - x.std()) / x.std() < 0.05


def test_bandpass_attenuates_out_of_band():
    in_band = _sine(10.0, amp=1.0)
    out_band = _sine(80.0, amp=1.0)
    mixed = in_band + out_band
    y = bandpass(mixed, 1.0, 40.0, FS)
    # The filtered signal should be much closer to the pure 10 Hz sine.
    assert np.std(y - in_band) < 0.2
    assert np.std(y - out_band) > np.std(y - in_band)


def test_bandpass_attenuates_dc():
    x = _sine(10.0) + 5.0  # offset
    y = bandpass(x, 1.0, 40.0, FS)
    assert abs(y.mean()) < 0.1


def test_basic_features_band_localization():
    feats = basic_features(_sine(10.0, n=512), FS)
    # Pure 10 Hz signal: nearly all relative power should land in the alpha band.
    assert feats["power_alpha"] > 0.7
    assert feats["power_delta"] < 0.1
    assert feats["power_beta"] < 0.1
    assert feats["power_gamma"] < 0.05


def test_sliding_windows_shape_and_content():
    x = np.arange(20)
    w = sliding_windows(x, window=5, hop=5)
    assert w.shape == (4, 5)
    np.testing.assert_array_equal(w[0], np.arange(0, 5))
    np.testing.assert_array_equal(w[-1], np.arange(15, 20))


def test_sliding_windows_rejects_2d():
    with pytest.raises(ValueError):
        sliding_windows(np.zeros((2, 10)), window=4, hop=4)


def test_pipeline_features_mode_shape_and_names():
    ds = simulate(SimConfig(n_patients=6, n_sessions=4, window=128, seed=0))
    out = preprocess(ds.signals, PipelineConfig(fs=FS, mode="features"))
    assert out.x.shape == (6, 4, len(FEATURE_NAMES))
    assert out.feature_names == FEATURE_NAMES
    assert out.x.dtype == np.float32
    # z-score per patient → mean of each feature within a patient should be ~0
    assert np.allclose(out.x.mean(axis=1), 0.0, atol=1e-5)


def test_pipeline_raw_mode_shape_and_normalization():
    ds = simulate(SimConfig(n_patients=5, n_sessions=3, window=64, seed=1))
    out = preprocess(ds.signals, PipelineConfig(fs=FS, mode="raw"))
    assert out.x.shape == ds.signals.shape
    assert out.feature_names is None
    # Per-patient z-score → mean across (sessions, window) per patient ~ 0, std ~ 1.
    flat = out.x.reshape(out.x.shape[0], -1)
    assert np.allclose(flat.mean(axis=1), 0.0, atol=1e-5)
    assert np.allclose(flat.std(axis=1), 1.0, atol=1e-5)


def test_pipeline_no_cross_patient_leakage():
    """Scaling one patient's signal must not affect any other patient's output."""
    ds = simulate(SimConfig(n_patients=4, n_sessions=3, window=64, seed=2))
    out_a = preprocess(ds.signals, PipelineConfig(fs=FS, mode="features"))

    perturbed = ds.signals.copy()
    perturbed[0] *= 1000.0  # explode patient 0
    out_b = preprocess(perturbed, PipelineConfig(fs=FS, mode="features"))

    np.testing.assert_allclose(out_a.x[1:], out_b.x[1:], atol=1e-4)


def test_preprocessing_domain_class_delegates():
    ds = simulate(SimConfig(n_patients=3, n_sessions=2, window=64, seed=4))
    pre = Preprocessing()
    out = pre.pipeline_dataset(ds.signals, fs=FS, mode="features")
    assert out.x.shape == (3, 2, len(FEATURE_NAMES))
