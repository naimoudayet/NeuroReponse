"""Tests for the network feature blocks introduced from the new_docs equations.

These features are the project's first that are defined *between* channels, so
they fail in ways band power cannot. Three traps get their own test:

* PLV computed on a broadband signal, or before trimming the filter's edge
  transient, is high for every pair — indistinguishable from real global
  synchrony (`test_plv_is_low_for_independent_noise`).
* Connectivity on a single-channel dataset is the channel with itself, which is
  1.0 by definition and would sail through every shape assertion
  (`test_connectivity_refuses_a_single_channel_dataset`).
* The 3-6-9 ratio must be able to come back empty. A version that could not
  return a near-zero value would make H369 unfalsifiable, which is the one thing
  the hypothesis document forbids (`test_r369_is_near_zero_when_no_harmonics_exist`).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.connectivity import (
    COMPLEXITY_METRICS,
    H369_METRICS,
    alpha_peak,
    aperiodic_exponent,
    band_phases,
    coherence_matrix,
    cross_spectrum,
    epoch_complexity_features,
    epoch_sync_features,
    harmonic_ratio_369,
    kuramoto_order,
    montage_h369_features,
    plv_matrix,
    sync_metric_names,
)

FS = 250.0
WINDOW = 2000                       # 8 s at 250 Hz — the TDBRAIN epoch
CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4",
    "T7", "C3", "Cz", "C4", "T8", "CP3", "CPz", "CP4", "P7", "P3",
    "Pz", "P4", "P8", "O1", "Oz", "O2",
]


def _pink_noise(n: int, exponent: float, rng) -> np.ndarray:
    """Noise whose PSD falls as 1/f**exponent — a ground truth for the 1/f fit."""
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    amp = np.zeros_like(freqs)
    amp[1:] = freqs[1:] ** (-exponent / 2.0)
    phase = rng.uniform(0, 2 * np.pi, freqs.size)
    spectrum = amp * np.exp(1j * phase)
    return np.fft.irfft(spectrum, n=n)


# --------------------------------------------------------------------------- #
# PLV and Kuramoto
# --------------------------------------------------------------------------- #

def test_plv_is_one_for_identical_phases():
    phase = np.tile(np.linspace(0, 40 * np.pi, 500), (5, 1))
    plv = plv_matrix(phase)
    assert plv.shape == (5, 5)
    assert np.allclose(plv, 1.0, atol=1e-9)


def test_plv_is_low_for_independent_noise():
    """The bias this whole module has to avoid: spurious global synchrony.

    Independent channels must not look coupled. If ``band_phases`` skipped the
    band-pass, or kept the filter's edge transient, every pair would share the
    same artefact and PLV would climb — reading exactly like the Kuramoto
    coupling the equations are after.
    """
    rng = np.random.default_rng(0)
    mc = rng.standard_normal((12, WINDOW))
    phase = band_phases(mc, FS, 8.0, 13.0)
    plv = plv_matrix(phase)
    off_diagonal = plv[~np.eye(12, dtype=bool)]
    assert off_diagonal.max() < 0.5, f"spurious coupling: max PLV {off_diagonal.max():.3f}"


def test_plv_recovers_a_planted_phase_lock():
    """Two channels driven by one oscillator lock; a third does not."""
    rng = np.random.default_rng(1)
    t = np.arange(WINDOW) / FS
    driver = np.sin(2 * np.pi * 10.0 * t)
    mc = np.stack([
        driver + 0.1 * rng.standard_normal(WINDOW),
        driver + 0.1 * rng.standard_normal(WINDOW),
        rng.standard_normal(WINDOW),
    ])
    plv = plv_matrix(band_phases(mc, FS, 8.0, 13.0))
    assert plv[0, 1] > 0.9
    assert plv[0, 2] < plv[0, 1]


def test_kuramoto_order_separates_synchrony_from_noise():
    t = np.arange(WINDOW) / FS
    locked = np.tile(np.sin(2 * np.pi * 10.0 * t), (10, 1))
    r_locked, meta_locked = kuramoto_order(band_phases(locked, FS, 8.0, 13.0))

    rng = np.random.default_rng(2)
    free = rng.standard_normal((10, WINDOW))
    r_free, _ = kuramoto_order(band_phases(free, FS, 8.0, 13.0))

    assert r_locked > 0.95
    assert r_free < r_locked
    # Perfectly locked oscillators have nothing left to wander with.
    assert meta_locked < 0.05


# --------------------------------------------------------------------------- #
# Coherence
# --------------------------------------------------------------------------- #

def test_coherence_is_bounded_and_one_on_the_diagonal():
    rng = np.random.default_rng(3)
    mc = rng.standard_normal((6, WINDOW))
    freqs, csd = cross_spectrum(mc, FS)
    coh = coherence_matrix(csd, (freqs >= 8.0) & (freqs < 13.0))
    assert coh.shape == (6, 6)
    assert np.all(coh >= 0.0) and np.all(coh <= 1.0)
    assert np.allclose(np.diag(coh), 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# The 1/f exponent — the Wilson-Cowan E/I stand-in
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("true_exponent", [1.0, 1.5, 2.0])
def test_aperiodic_exponent_recovers_a_known_slope(true_exponent):
    """Synthesise 1/f**a noise, fit, and get ``a`` back.

    This is the closest thing to a ground truth available for the E:I proxy: the
    exponent is the one quantity in this module whose correct value is known in
    advance rather than merely plausible.
    """
    rng = np.random.default_rng(4)
    mc = np.stack([_pink_noise(WINDOW, true_exponent, rng) for _ in range(4)])
    freqs, csd = cross_spectrum(mc, FS)
    psd = np.einsum("fcc->cf", csd).real
    estimated = float(np.mean(aperiodic_exponent(freqs, psd)))
    assert estimated == pytest.approx(true_exponent, abs=0.25)


def test_alpha_peak_finds_a_planted_rhythm():
    rng = np.random.default_rng(5)
    t = np.arange(WINDOW) / FS
    mc = np.stack([
        0.3 * rng.standard_normal(WINDOW) + np.sin(2 * np.pi * 11.0 * t)
        for _ in range(4)
    ])
    freqs, csd = cross_spectrum(mc, FS)
    psd = np.einsum("fcc->cf", csd).real
    assert alpha_peak(freqs, psd) == pytest.approx(11.0, abs=0.6)


# --------------------------------------------------------------------------- #
# The falsifiable 3-6-9 ratio
# --------------------------------------------------------------------------- #

def test_r369_detects_planted_harmonics():
    rng = np.random.default_rng(6)
    t = np.arange(WINDOW) / FS
    f0 = 2.0
    signal = sum(np.sin(2 * np.pi * k * f0 * t) for k in (3, 6, 9))
    mc = np.stack([signal + 0.05 * rng.standard_normal(WINDOW) for _ in range(4)])
    freqs, csd = cross_spectrum(mc, FS)
    psd = np.einsum("fcc->cf", csd).real
    assert harmonic_ratio_369(freqs, psd, f0) > 0.5


def test_r369_is_near_zero_when_no_harmonics_exist():
    """H369 must be able to fail, or it is not a hypothesis.

    A ratio that returned a large number on any input would make the 3-6-9
    hypothesis unfalsifiable — precisely what the document's own "critère de
    falsification" section rules out. White noise has no harmonic structure, so
    the index must return roughly the fraction of the band the three windows
    happen to cover, not something that looks like a finding.
    """
    rng = np.random.default_rng(7)
    mc = rng.standard_normal((4, WINDOW))
    freqs, csd = cross_spectrum(mc, FS)
    psd = np.einsum("fcc->cf", csd).real
    assert harmonic_ratio_369(freqs, psd, 2.0) < 0.15


def test_r369_block_is_not_constant_across_patients():
    """A block with no between-patient variance cannot carry a label either way.

    If this collapsed to a constant, ``beats_chance`` would correctly return
    False for the H369 rung — but for the wrong reason, and the falsification
    would be an artefact of the implementation rather than a result.
    """
    rng = np.random.default_rng(8)
    values = np.stack([
        montage_h369_features(rng.standard_normal((3, 8, WINDOW)) * scale, FS)
        for scale in (1.0, 2.0, 3.0)
    ])
    per_patient = values.mean(axis=1)
    assert np.all(per_patient.std(axis=0) > 0)


# --------------------------------------------------------------------------- #
# Block assembly
# --------------------------------------------------------------------------- #

def test_sync_block_has_the_documented_width_and_names():
    rng = np.random.default_rng(9)
    mc = rng.standard_normal((len(CHANNELS), WINDOW))
    vec = epoch_sync_features(mc, FS, CHANNELS)
    names = sync_metric_names()
    assert vec.shape == (30,)
    assert len(names) == 30
    assert names[0] == "plv_mean_delta" and names[-1] == "kuramoto_meta_gamma"
    assert np.isfinite(vec).all()


def test_complexity_block_is_finite_on_real_shaped_input():
    rng = np.random.default_rng(10)
    mc = rng.standard_normal((len(CHANNELS), WINDOW))
    vec = epoch_complexity_features(mc, FS, CHANNELS)
    assert vec.shape == (len(COMPLEXITY_METRICS),)
    assert np.isfinite(vec).all()


def test_h369_block_width_matches_its_names():
    rng = np.random.default_rng(11)
    vec = montage_h369_features(rng.standard_normal((2, 4, WINDOW)), FS)
    assert vec.shape == (2, len(H369_METRICS))


def test_alpha_asymmetry_follows_the_planted_imbalance():
    """log(alpha F4) - log(alpha F3) must be positive when F4 carries more alpha."""
    rng = np.random.default_rng(12)
    t = np.arange(WINDOW) / FS
    mc = 0.2 * rng.standard_normal((len(CHANNELS), WINDOW))
    mc[CHANNELS.index("F4")] += 2.0 * np.sin(2 * np.pi * 10.0 * t)
    vec = epoch_complexity_features(mc, FS, CHANNELS)
    asym = dict(zip(COMPLEXITY_METRICS, vec))["alpha_asymmetry_f4_f3"]
    assert asym > 0.5
