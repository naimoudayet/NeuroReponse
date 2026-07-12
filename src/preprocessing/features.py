from __future__ import annotations

import numpy as np
from scipy.signal import welch

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def basic_features(x: np.ndarray, fs: float) -> dict[str, float]:
    feats: dict[str, float] = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": float(np.sqrt(np.mean(x**2))),
    }
    nperseg = min(256, len(x))
    freqs, psd = welch(x, fs=fs, nperseg=nperseg)
    total = float(np.trapezoid(psd, freqs)) or 1.0
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        feats[f"power_{name}"] = float(np.trapezoid(psd[mask], freqs[mask]) / total)
    return feats


def _peak_in_window(x: np.ndarray, t: np.ndarray, lo: float, hi: float, kind: str) -> tuple[float, float]:
    """Return (amplitude, latency_s) of the extremum in the [lo, hi] s window.

    kind="neg" finds the trough (N100), kind="pos" finds the peak (P300).
    Amplitude is returned as a positive magnitude.
    """
    mask = (t >= lo) & (t < hi)
    if not mask.any():                       # window falls outside a short signal
        mask = np.ones_like(t, dtype=bool)
    seg = x[mask]
    idx = int(np.argmin(seg)) if kind == "neg" else int(np.argmax(seg))
    latency = float(t[mask][idx])
    amp = float(abs(seg[idx]))
    return amp, latency


def erp_features(x: np.ndarray, fs: float) -> dict[str, float]:
    """Cognitive-channel (ERP) features: N100 / P300 amplitude + latency and rectified area."""
    t = np.arange(len(x)) / fs
    n100_amp, n100_lat = _peak_in_window(x, t, 0.08, 0.15, "neg")
    p300_amp, p300_lat = _peak_in_window(x, t, 0.25, 0.40, "pos")
    return {
        "erp_n100_amp": n100_amp,
        "erp_n100_lat": n100_lat,
        "erp_p300_amp": p300_amp,
        "erp_p300_lat": p300_lat,
        "erp_auc": float(np.trapezoid(np.abs(x), t)),
    }


def hrv_features(rr: np.ndarray) -> dict[str, float]:
    """Autonomic-channel (ECG) features from an RR-interval tachogram in seconds.

    SDNN / RMSSD / pNN50 are standard time-domain HRV metrics; LF/HF is a coarse
    spectral proxy computed on the (approximately evenly sampled) RR series.
    """
    rr = np.asarray(rr, dtype=np.float64)
    mean_rr = float(np.mean(rr)) if rr.size else 1.0
    diffs = np.diff(rr) if rr.size > 1 else np.array([0.0])

    sdnn = float(np.std(rr) * 1000.0)
    rmssd = float(np.sqrt(np.mean(diffs**2)) * 1000.0)
    pnn50 = float(np.mean(np.abs(diffs) * 1000.0 > 50.0)) if diffs.size else 0.0
    hr_mean = float(60.0 / mean_rr) if mean_rr > 0 else 0.0

    lf_hf = 0.0
    if rr.size >= 16 and mean_rr > 0:
        fs_rr = 1.0 / mean_rr
        freqs, psd = welch(rr - np.mean(rr), fs=fs_rr, nperseg=min(rr.size, 32))
        lf = float(np.trapezoid(psd[(freqs >= 0.04) & (freqs < 0.15)],
                                freqs[(freqs >= 0.04) & (freqs < 0.15)]))
        hf = float(np.trapezoid(psd[(freqs >= 0.15) & (freqs < 0.40)],
                                freqs[(freqs >= 0.15) & (freqs < 0.40)]))
        lf_hf = lf / hf if hf > 0 else 0.0

    return {
        "hrv_hr_mean": hr_mean,
        "hrv_sdnn": sdnn,
        "hrv_rmssd": rmssd,
        "hrv_pnn50": pnn50,
        "hrv_lf_hf": lf_hf,
    }
