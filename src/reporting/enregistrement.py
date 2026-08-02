"""Read a clinician-supplied recording into the montage + tachogram the model needs.

A loop iteration under TDBRAIN starts from a real acquisition: a multi-channel
resting EEG with an ECG lead. Rather than reimplement that parsing, this delegates
to :mod:`src.data.tdbrain` — the same functions that built the training set — so an
uploaded file gets the identical notch/band-pass, the identical resample to the
contract's rate, and the identical R-peak detection. Anything else would train on
one preprocessing and predict on another.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def lire_enregistrement(
    path: Path,
    channels: tuple[str, ...] | list[str],
    fs_cible: float,
    ecg_channel: str | None = "Erbs",
    notch_hz: float | None = 50.0,
    bandpass_hz: tuple[float, float] | None = (1.0, 45.0),
    n_rr: int = 64,
) -> tuple[dict[str, np.ndarray], np.ndarray | None, float]:
    """Parse a ``.bdf``/``.csv`` recording -> ``(montage, tachogram, fs)``.

    ``montage`` maps each requested channel name to its full trace, resampled to
    ``fs_cible``. ``tachogram`` holds RR intervals in seconds, or ``None`` when no
    usable autonomic lead is present.

    ``n_rr`` must match the checkpoint's ``FeatureContract.n_rr``. The tachogram is
    fitted to that length **here**, exactly as ``load_tdbrain`` does, because HRV is
    computed from the fitted series: SDNN over 140 beats is not SDNN over the first
    64, so skipping this step trains on one autonomic feature and predicts on
    another. ``tests/test_boucle_tdbrain.py`` pins the two paths together.

    Raises ``KeyError`` when a required channel is absent and ``ValueError`` when
    the file cannot be parsed — both carry the offending detail so the caller can
    tell the clinician what to fix.
    """
    from ..data.tdbrain import (
        SOURCE_FS,
        _read_recording,
        _resample,
        _tachogram,
        detect_rr_intervals,
    )

    chans = tuple(channels)
    eeg, ecg_raw, ecg_fs = _read_recording(
        Path(path), chans, ecg_channel, notch_hz, bandpass_hz
    )
    eeg = _resample(eeg, SOURCE_FS, float(fs_cible))

    montage = {
        c: np.ascontiguousarray(eeg[i], dtype=np.float32) for i, c in enumerate(chans)
    }

    tach = None
    if ecg_channel and ecg_raw.size:
        # Detect at the native rate: the resample to fs_cible blunts R-peak timing.
        rr = detect_rr_intervals(ecg_raw, ecg_fs)
        if rr.size:
            tach = _tachogram(rr, n_rr)
    return montage, tach, float(fs_cible)


def generer_enregistrement_demo(
    channels: tuple[str, ...] | list[str],
    fs: float,
    duree_s: float,
    seed: int = 0,
    n_rr: int = 64,
    alpha: float = 1.5,
    bpm: float = 68.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Synthesise a montage + tachogram so the loop can be exercised without hardware.

    **Not medical data.** It reuses the same generators as the test fixture so the
    shapes and units are right, and exists only to let someone walk the workflow
    end-to-end before an amplifier is connected. ``alpha`` and ``bpm`` are exposed
    so successive demo sessions can be made to differ.
    """
    from ..data.tdbrain import (
        SOURCE_FS,
        _synthetic_qrs,
        _tachogram,
        detect_rr_intervals,
    )

    rng = np.random.default_rng(seed)
    n = int(round(duree_s * fs))
    t = np.arange(n) / fs
    montage = {}
    for c in channels:
        theta = 1.0 * np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 2 * np.pi))
        a = alpha * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))
        beta = 0.5 * np.sin(2 * np.pi * 20.0 * t + rng.uniform(0, 2 * np.pi))
        montage[c] = (theta + a + beta + 0.6 * rng.standard_normal(n)).astype(np.float32)

    # Build the ECG at the source rate, then detect exactly as the real path does.
    ecg = _synthetic_qrs(int(round(duree_s * SOURCE_FS)), SOURCE_FS, bpm, rng)
    rr = detect_rr_intervals(ecg, SOURCE_FS)
    return montage, _tachogram(rr, n_rr), float(fs)


def duree_secondes(montage: dict[str, np.ndarray], fs: float) -> float:
    """Recording length in seconds, for reporting against the contract's minimum."""
    if not montage or fs <= 0:
        return 0.0
    return float(next(iter(montage.values())).size) / float(fs)
