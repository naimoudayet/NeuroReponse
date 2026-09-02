"""Network-level EEG features: synchronisation, connectivity, complexity, harmonics.

Everything in this module answers one question the reference study (Arteaga et
al., PMC12981298) never asks: **is there information in how the channels relate
to each other, rather than in how much power each one carries on its own?**
Their model, and this project's original TDBRAIN arm, both see nothing but
relative band power per channel — 130 numbers that are individually blind to
phase. The equation blocks in ``new_docs/`` (Kuramoto coupling, PLV, spectral
coherence, the 1/f slope standing in for the Wilson–Cowan E/I state) all live at
that missing level, so they are the part of those documents that is both
implementable on TDBRAIN and capable of adding something.

Three blocks, and each one is deliberately **small**:

``sync``       6 features x 5 bands = 30 — PLV, coherence, Kuramoto order.
``complexity`` 6 — spectral entropy, aperiodic exponent, frontal alpha asymmetry.
``h369``       4 — the falsifiable Tesla 3-6-9 harmonic ratio.

Compactness is a measured requirement, not taste. This project has already
established that 139 features beat 4 features *downwards* on 132 patients: the
clinical model (4 columns) outscores the multimodal one (139) on both cohorts,
because 135 uninformative columns dilute the informative ones. A connectivity
block that emitted one column per channel pair would add 325 x 5 = 1625 columns
to a 132-patient cohort and could only make that worse. So every quantity here is
already aggregated over channels or over anatomically motivated channel groups
before it reaches the model.

The aggregation regions are not arbitrary either. rTMS in this cohort targets the
DLPFC, so the frontal group is the one where a stimulation-related effect should
appear; the left/right split exists because the two protocols stimulate opposite
hemispheres.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, get_window, hilbert, sosfiltfilt

from .features import BANDS

# Anatomical groups over the canonical 26-channel montage. Membership is by
# name, never by row index — the same rule that keeps the database's row order
# from silently permuting a feature vector elsewhere in this project.
FRONTAL: tuple[str, ...] = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4",
)
# 10-20 convention: odd = left, even = right, "z" = midline (in neither group).
LEFT_SUFFIX = ("1", "3", "5", "7", "9")
RIGHT_SUFFIX = ("2", "4", "6", "8")

# Frontal alpha asymmetry is computed on this pair. F3/F4 is the pair the
# depression literature uses, and F4 is inside the protocol-2 stimulation site.
ASYMMETRY_PAIR = ("F3", "F4")

SYNC_METRICS: tuple[str, ...] = (
    "plv_mean",           # mean phase-locking value over every channel pair
    "plv_frontal",        # ... restricted to frontal-frontal pairs (the rTMS target)
    "plv_inter",          # ... restricted to left-right pairs (interhemispheric)
    "coh_mean",           # mean magnitude-squared coherence over every pair
    "kuramoto_r",         # global Kuramoto order parameter, time-averaged
    "kuramoto_meta",      # its standard deviation over time = metastability
)

COMPLEXITY_METRICS: tuple[str, ...] = (
    "spectral_entropy_mean",
    "spectral_entropy_frontal",
    "aperiodic_exponent_mean",       # 1/f slope; the E:I balance proxy
    "aperiodic_exponent_frontal",
    "alpha_asymmetry_f4_f3",         # log(alpha F4) - log(alpha F3)
    "alpha_peak_hz",                 # individual alpha frequency
)

# Fundamental frequencies the 3-6-9 ratio is evaluated at. 1.0 Hz puts the
# harmonics at 3/6/9 Hz (delta-theta-alpha edge), 2.0 Hz at 6/12/18 and 3.0 Hz
# at 9/18/27. `iaf` uses the patient's own alpha peak divided by three, which is
# the only variant with any physiological motivation at all.
H369_F0: tuple[float, ...] = (1.0, 2.0, 3.0)

H369_METRICS: tuple[str, ...] = tuple(
    [f"r369_f0_{f0:g}" for f0 in H369_F0] + ["r369_iaf"]
)

# Half-bandwidth used to integrate power around each harmonic. Wide enough to
# survive the 0.125 Hz frequency resolution of an 8 s epoch, narrow enough that
# 3f0 / 6f0 / 9f0 do not overlap for any f0 in H369_F0.
H369_HALFWIDTH_HZ = 0.5

_ALPHA_LO, _ALPHA_HI = BANDS["alpha"]
# The 1/f fit window. Starts above the high-pass corner's transition band and
# stops below the 50 Hz notch, so neither filter's own shape is fitted as if it
# were neural.
_APERIODIC_RANGE = (2.0, 40.0)


# --------------------------------------------------------------------------- #
# Spectral primitives
# --------------------------------------------------------------------------- #

def _segments(x: np.ndarray, nperseg: int, noverlap: int) -> np.ndarray:
    """``(n_channels, window)`` -> ``(n_channels, n_segments, nperseg)``, Hann-tapered.

    Written out rather than delegated to ``scipy.signal.welch`` because the
    cross-spectral density needs the *complex* per-segment FFT of every channel
    at once. Calling ``scipy.signal.csd`` per pair would recompute the same 26
    FFTs 325 times per band per epoch.
    """
    n = x.shape[-1]
    if n < nperseg:
        nperseg = n
        noverlap = 0
    step = max(nperseg - noverlap, 1)
    starts = range(0, n - nperseg + 1, step)
    seg = np.stack([x[..., s:s + nperseg] for s in starts], axis=-2)
    win = get_window("hann", nperseg)
    return (seg - seg.mean(axis=-1, keepdims=True)) * win


def cross_spectrum(mc: np.ndarray, fs: float, nperseg: int | None = None):
    """Welch cross-spectral density for every channel pair, in one pass.

    Returns ``(freqs, csd)`` with ``csd`` of shape ``(n_freqs, n_ch, n_ch)``:
    the Hermitian matrix S(f) whose diagonal is each channel's own PSD and whose
    off-diagonal entry ``S_jk(f)`` is the complex cross-spectrum feeding the
    coherence C_xy(f) = |S_xy|^2 / (S_xx S_yy) of the documents' equation 5.
    """
    n = mc.shape[-1]
    nperseg = int(nperseg or min(512, n))
    seg = _segments(np.asarray(mc, dtype=np.float64), nperseg, nperseg // 2)
    spec = np.fft.rfft(seg, axis=-1)                   # (n_ch, n_seg, n_freq)
    freqs = np.fft.rfftfreq(seg.shape[-1], d=1.0 / fs)
    # Average the outer product over segments: S[f] = <x(f) x(f)^H>_segments.
    csd = np.einsum("csf,dsf->fcd", spec, spec.conj()) / spec.shape[1]
    return freqs, csd


def _band_mask(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():                       # band narrower than the resolution
        mask = np.zeros_like(freqs, dtype=bool)
        mask[np.argmin(np.abs(freqs - (lo + hi) / 2))] = True
    return mask


def coherence_matrix(csd: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Band-averaged magnitude-squared coherence, ``(n_ch, n_ch)``.

    Coherence is averaged **after** forming C_xy(f) at each frequency, not by
    averaging the cross-spectra first: the latter is a different (and much more
    optimistic) quantity, since it lets a strong narrow-band component dominate
    the whole band's phase consistency.
    """
    s = csd[mask]                                        # (n_f, n_ch, n_ch)
    auto = np.einsum("fcc->fc", s).real
    denom = auto[:, :, None] * auto[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.abs(s) ** 2 / denom
    coh = np.where(np.isfinite(coh), coh, 0.0)
    return np.clip(coh.mean(axis=0), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Phase primitives — Kuramoto / PLV
# --------------------------------------------------------------------------- #

def band_phases(mc: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """Instantaneous phase per channel in ``[lo, hi]`` Hz, ``(n_ch, window)``.

    Band-pass then Hilbert, which is the only order that makes the analytic
    phase meaningful: the Hilbert transform of a broadband signal has no single
    well-defined instantaneous frequency, so a phase read off it is not the
    theta_i of the Kuramoto equation.

    The edges are trimmed by a quarter of the slowest period on each side. A
    zero-phase ``sosfiltfilt`` leaves a transient there whose phase is the
    filter's, not the brain's, and PLV over a shared artefact is high for every
    pair — a bias that would look exactly like global synchrony.
    """
    nyq = fs / 2.0
    lo_n, hi_n = max(lo / nyq, 1e-4), min(hi / nyq, 0.99)
    sos = butter(4, [lo_n, hi_n], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, np.asarray(mc, dtype=np.float64), axis=-1)
    phase = np.angle(hilbert(filtered, axis=-1))
    trim = int(round(0.25 * fs / max(lo, 0.5)))
    if 2 * trim < phase.shape[-1] - 8:
        phase = phase[..., trim:phase.shape[-1] - trim]
    return phase


def plv_matrix(phase: np.ndarray) -> np.ndarray:
    """Phase-locking value for every channel pair, ``(n_ch, n_ch)``.

    PLV_jk = |<exp(i(phi_j - phi_k))>_t|, the documents' equation 5 written as
    one matrix product: with z = exp(i phi), the average of z_j conj(z_k) over
    time *is* ``z @ z.conj().T / T``.
    """
    z = np.exp(1j * phase)
    return np.abs(z @ z.conj().T) / phase.shape[-1]


def kuramoto_order(phase: np.ndarray) -> tuple[float, float]:
    """``(mean R, std R)`` — global synchrony and metastability.

    R(t) = |(1/N) sum_j exp(i theta_j(t))| is the order parameter of the
    Kuramoto model the documents specify. Its *variance* over time is reported
    alongside it because a brain pinned at constant R is pathological in either
    direction: R = 1 is a seizure, R = 0 is noise, and healthy cortex wanders.
    Metastability is that wandering, and it is the half of the Kuramoto picture a
    time-averaged R throws away.
    """
    r = np.abs(np.exp(1j * phase).mean(axis=0))
    return float(r.mean()), float(r.std())


# --------------------------------------------------------------------------- #
# Complexity primitives
# --------------------------------------------------------------------------- #

def spectral_entropy(psd: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalised Shannon entropy of the PSD, one value per channel.

    1.0 is a flat (maximally unpredictable) spectrum, 0.0 a single peak.
    """
    p = np.asarray(psd)[:, mask]
    total = p.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(total > 0, p / total, 0.0)
        ent = -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=-1)
    return ent / np.log(max(int(mask.sum()), 2))


def aperiodic_exponent(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Slope of log PSD against log f, returned as a positive exponent per channel.

    This is the module's stand-in for the Wilson-Cowan state ``[E(t), I(t)]`` the
    documents put at the centre of the latent vector. Fitting Wilson-Cowan itself
    to a resting recording is a parameter-estimation problem this cohort cannot
    support — but the 1/f slope of the power spectrum is an established proxy for
    the *ratio* of the two (Gao, Peterson & Voytek 2017), and the ratio is what
    the equations actually use. One number per channel, estimated from data we
    already have, instead of four unidentifiable ones.

    Fitted over 2-40 Hz to stay clear of the 0.01 Hz high-pass transition and the
    50 Hz notch.
    """
    lo, hi = _APERIODIC_RANGE
    mask = (freqs >= lo) & (freqs <= hi) & (freqs > 0)
    if mask.sum() < 3:
        return np.full(np.asarray(psd).shape[0], np.nan)
    lf = np.log10(freqs[mask])
    lp = np.log10(np.maximum(np.asarray(psd)[:, mask], 1e-30))
    design = np.column_stack([np.ones_like(lf), lf])
    coef, *_ = np.linalg.lstsq(design, lp.T, rcond=None)
    return -coef[1]                                   # positive = steeper 1/f


def alpha_peak(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Individual alpha frequency: the argmax of the montage-mean PSD in 8-13 Hz."""
    mask = _band_mask(freqs, _ALPHA_LO, _ALPHA_HI)
    mean_psd = np.asarray(psd)[:, mask].mean(axis=0)
    return float(freqs[mask][int(np.argmax(mean_psd))])


# --------------------------------------------------------------------------- #
# The 3-6-9 harmonic ratio (HYPO4-369)
# --------------------------------------------------------------------------- #

def harmonic_ratio_369(
    freqs: np.ndarray,
    psd: np.ndarray,
    f0: float,
    halfwidth: float = H369_HALFWIDTH_HZ,
) -> float:
    """R369 = [P(3f0) + P(6f0) + P(9f0)] / P_total, montage-averaged.

    Implemented exactly as the document writes it, including the part that makes
    it falsifiable: nothing here assumes those frequencies are special. Power is
    integrated in a +/- ``halfwidth`` window around each harmonic and divided by
    the total power over the analysed band, so the result is a bounded fraction
    that a permutation test can null out. If 3-6-9 carries no information, this
    returns a number that correlates with nothing — which is the outcome the
    hypothesis has to survive.
    """
    psd = np.asarray(psd)
    total = float(psd.sum())
    if total <= 0 or not np.isfinite(total) or not np.isfinite(f0) or f0 <= 0:
        return 0.0
    acc = 0.0
    for k in (3, 6, 9):
        target = k * f0
        if target > freqs[-1]:
            continue
        m = (freqs >= target - halfwidth) & (freqs <= target + halfwidth)
        acc += float(psd[:, m].sum())
    return acc / total


# --------------------------------------------------------------------------- #
# Epoch -> feature vector
# --------------------------------------------------------------------------- #

def _group_mask(channels, keep) -> np.ndarray:
    return np.array([c in keep for c in channels], dtype=bool)


def _hemisphere_masks(channels) -> tuple[np.ndarray, np.ndarray]:
    left = np.array([c[-1] in LEFT_SUFFIX for c in channels], dtype=bool)
    right = np.array([c[-1] in RIGHT_SUFFIX for c in channels], dtype=bool)
    return left, right


def _pair_mean(matrix: np.ndarray, rows: np.ndarray, cols: np.ndarray,
               symmetric: bool) -> float:
    """Mean of a connectivity matrix over a set of pairs, self-pairs excluded.

    ``symmetric`` selects within-group pairs (upper triangle of a square block);
    otherwise every ``rows x cols`` pair is taken, which is what an
    interhemispheric average wants.
    """
    if not rows.any() or not cols.any():
        return float("nan")
    block = matrix[np.ix_(rows, cols)]
    if symmetric:
        iu = np.triu_indices(block.shape[0], k=1)
        vals = block[iu]
    else:
        vals = block.ravel()
    return float(vals.mean()) if vals.size else float("nan")


def sync_metric_names(bands=None) -> tuple[str, ...]:
    """Column names for the ``sync`` block: band-major, metric-minor."""
    return tuple(
        f"{metric}_{band}"
        for band in (bands or BANDS)
        for metric in SYNC_METRICS
    )


def epoch_sync_features(mc: np.ndarray, fs: float, channels) -> np.ndarray:
    """One epoch ``(n_channels, window)`` -> 30 synchronisation features."""
    channels = list(channels)
    frontal = _group_mask(channels, set(FRONTAL))
    left, right = _hemisphere_masks(channels)
    everything = np.ones(len(channels), dtype=bool)

    freqs, csd = cross_spectrum(mc, fs)
    out: list[float] = []
    for lo, hi in BANDS.values():
        phase = band_phases(mc, fs, lo, hi)
        plv = plv_matrix(phase)
        coh = coherence_matrix(csd, _band_mask(freqs, lo, hi))
        r_mean, r_std = kuramoto_order(phase)
        out.extend([
            _pair_mean(plv, everything, everything, symmetric=True),
            _pair_mean(plv, frontal, frontal, symmetric=True),
            _pair_mean(plv, left, right, symmetric=False),
            _pair_mean(coh, everything, everything, symmetric=True),
            r_mean,
            r_std,
        ])
    return np.asarray(out, dtype=np.float32)


def epoch_complexity_features(mc: np.ndarray, fs: float, channels) -> np.ndarray:
    """One epoch -> 6 complexity / asymmetry features."""
    channels = list(channels)
    freqs, csd = cross_spectrum(mc, fs)
    psd = np.einsum("fcc->cf", csd).real                  # (n_ch, n_freq)
    frontal = _group_mask(channels, set(FRONTAL))

    ent = spectral_entropy(psd, freqs > 0)
    slope = aperiodic_exponent(freqs, psd)

    alpha = _band_mask(freqs, _ALPHA_LO, _ALPHA_HI)
    asym = float("nan")
    f3, f4 = ASYMMETRY_PAIR
    if f3 in channels and f4 in channels:
        p3 = float(psd[channels.index(f3), alpha].sum())
        p4 = float(psd[channels.index(f4), alpha].sum())
        if p3 > 0 and p4 > 0:
            asym = float(np.log(p4) - np.log(p3))

    return np.asarray([
        float(ent.mean()),
        float(ent[frontal].mean()) if frontal.any() else float("nan"),
        float(np.nanmean(slope)),
        float(np.nanmean(slope[frontal])) if frontal.any() else float("nan"),
        asym,
        alpha_peak(freqs, psd),
    ], dtype=np.float32)


def epoch_h369_features(mc: np.ndarray, fs: float) -> np.ndarray:
    """One epoch -> the 3-6-9 harmonic ratios (three fixed f0 plus IAF/3)."""
    freqs, csd = cross_spectrum(mc, fs)
    psd = np.einsum("fcc->cf", csd).real
    vals = [harmonic_ratio_369(freqs, psd, f0) for f0 in H369_F0]
    vals.append(harmonic_ratio_369(freqs, psd, alpha_peak(freqs, psd) / 3.0))
    return np.asarray(vals, dtype=np.float32)


def _stack_epochs(fn, mc: np.ndarray, *args) -> np.ndarray:
    """Apply a per-epoch extractor over ``(n_epochs, n_channels, window)``."""
    if mc.ndim != 3:
        raise ValueError(f"expected (n_epochs, n_channels, window); got {mc.shape}")
    return np.stack([fn(mc[s], *args) for s in range(mc.shape[0])]).astype(np.float32)


def montage_sync_features(mc: np.ndarray, fs: float, channels) -> np.ndarray:
    """``(n_epochs, n_ch, window)`` -> ``(n_epochs, 30)``."""
    return _stack_epochs(epoch_sync_features, mc, fs, channels)


def montage_complexity_features(mc: np.ndarray, fs: float, channels) -> np.ndarray:
    """``(n_epochs, n_ch, window)`` -> ``(n_epochs, 6)``."""
    return _stack_epochs(epoch_complexity_features, mc, fs, channels)


def montage_h369_features(mc: np.ndarray, fs: float) -> np.ndarray:
    """``(n_epochs, n_ch, window)`` -> ``(n_epochs, 4)``."""
    return _stack_epochs(epoch_h369_features, mc, fs)
