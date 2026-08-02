"""Real-data loader for the TDBRAIN rTMS-in-MDD cohort.

Maps the public TDBRAIN database (https://brainclinics.com/resources/) onto the
project's :class:`~src.data.loader.LoadedDataset` contract, so the existing LSTM
+ patient-wise cross-validation pipeline can learn from *real* EEG instead of the
simulator. See ``docs/tdbrain.md`` for how to obtain the data and the assumptions
this parser makes about the on-disk layout.

Reality vs. the simulator (this is why the mapping is not 1:1):
  * **EEG only** — TDBRAIN carries no evoked-potential or clean autonomic channel
    aligned to treatment, so ``erp`` and ``ecg`` stay ``None``. The multimodal
    (ERP + ECG) story remains the simulated NPDT track.
  * **Baseline only** — one pre-treatment resting recording per patient, so there
    is no rTMS-session trajectory. The LSTM's sequence axis is filled instead by
    *epochs* of the ~2-min recording (``snapshot=True`` collapses it to one window
    for the non-temporal baseline classifier).
  * **Label = binary responder** — ``>=50%`` BDI-II reduction pre->post; the raw
    delta and percentage are kept in ``metadata`` for an optional regression later.
  * The reference study (PMC12981298) found only **eyes-open** significant, hence
    ``condition="EO"`` by default, and modelled the two rTMS **protocols
    separately** (1 = 10 Hz L-DLPFC, 2 = 1 Hz R-DLPFC) — ``protocols`` selects them.

Real recordings are BioSemi **BDF** files (``*_eeg.bdf``), read with :mod:`mne`'s
self-contained BDF reader; the older CSV path is kept for headerless/derivative
exports and for the synthetic fixture (which can emit either ``.csv`` or ``.bdf``).

This module never ships patient data (CLAUDE.md). Point ``TDBRAINConfig.root`` at
your local TDBRAIN copy; :func:`make_synthetic_tdbrain` writes a tiny fake tree in
the same format for tests and for exercising the pipeline before you download it.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from ..preprocessing.features import BANDS, basic_features
from .loader import LoadedDataset

# 26-channel 10-10 montage, in the order given by the TDBRAIN data descriptor
# (Scientific Data, PMC9198070, Table 3).
TDBRAIN_CHANNELS_26: tuple[str, ...] = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4",
    "T7", "C3", "Cz", "C4", "T8", "CP3", "CPz", "CP4", "P7", "P3",
    "Pz", "P4", "P8", "O1", "Oz", "O2",
)

# The descriptor lists some channels with an older 10-20 alias.
_CHANNEL_ALIASES: dict[str, tuple[str, ...]] = {
    "P7": ("T5",),
    "P8": ("T6",),
}

SOURCE_FS: float = 500.0  # TDBRAIN native sampling rate


@dataclass
class TDBRAINConfig:
    """Where the data lives and how to turn it into the pipeline tensor.

    Column names default to ``None`` and are auto-resolved case-insensitively
    against a list of candidates (see :func:`_resolve_column`). If your download's
    ``participants.tsv`` uses different headers, set them explicitly.
    """

    root: Path
    metadata_path: Path | None = None           # participants table; defaults to root/participants.tsv
    condition: str = "EO"                       # "EO" (eyes-open) or "EC" (eyes-closed)
    channels: tuple[str, ...] = TDBRAIN_CHANNELS_26
    representative_channel: str = "Pz"          # single channel exposed as `signals` for the app

    # Autonomic channel. Every TDBRAIN recording carries an ECG lead at Erb's point
    # alongside the EEG montage, so HRV comes free with the same BDF read. Set to
    # None to skip it (the CSV fixture path has no ECG and skips it automatically).
    ecg_channel: str | None = "Erbs"
    ecg_bandpass_hz: tuple[float, float] | None = (5.0, 20.0)   # QRS band for R-peak detection
    n_rr: int = 64                              # RR intervals kept per patient (tachogram length)
    n_epochs: int = 8
    epoch_seconds: float = 8.0
    target_fs: float = 250.0                    # PMC12981298 downsampled 500 -> 250
    responder_threshold: float = 0.5            # >=50% BDI reduction = responder
    protocols: tuple[int, ...] = (1, 2)
    indication: str = "MDD"
    session: int = 1                            # baseline session index
    snapshot: bool = False                      # collapse epochs to one whole-recording window
    task: str = "response"                      # "response" (rTMS responder) or "diagnosis" (MDD vs control)
    control_indication: str = "HEALTHY"         # negative class when task="diagnosis"

    # Preprocessing for real BDF recordings (raw TDBRAIN is unfiltered). Ignored
    # for CSV input, which is assumed already clean. Set either to None to disable.
    notch_hz: float | None = 50.0               # power-line notch + harmonics (EU=50, US=60)
    bandpass_hz: tuple[float, float] | None = (1.0, 45.0)   # (high-pass, low-pass) in Hz

    # participants.tsv column resolution (auto-detected when None).
    col_id: str | None = None
    col_indication: str | None = None
    col_bdi_pre: str | None = None
    col_bdi_post: str | None = None
    col_protocol: str | None = None


def _demographics(row, cols: dict[str, str | None]) -> dict:
    """Pull age/gender out of a participants row when those columns exist.

    Both are optional and best-effort: an unparseable or absent value yields
    ``None`` rather than excluding the patient.
    """
    out: dict = {}
    if cols.get("age"):
        age = pd.to_numeric(row[cols["age"]], errors="coerce")
        out["age"] = float(age) if np.isfinite(age) else None
    if cols.get("gender"):
        gender = str(row[cols["gender"]] or "").strip()
        out["gender"] = gender or None
    return out


@dataclass
class _Skipped:
    reasons: dict[str, int] = field(default_factory=dict)

    def add(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


# --------------------------------------------------------------------------- #
# participants.tsv parsing
# --------------------------------------------------------------------------- #

_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "id": ("participants_id", "participant_id", "participants_ID", "id", "subject"),
    "indication": ("indication", "diagnosis", "dx", "group"),
    "bdi_pre": ("bdi_pre", "bdi_baseline", "bdi.pre", "bdipre", "bdi_1", "bdi_t0"),
    "bdi_post": ("bdi_post", "bdi_end", "bdi.post", "bdipost", "bdi_2", "bdi_t1"),
    "protocol": ("rtms_protocol", "protocol", "rtmsprotocol", "treatment_protocol", "protocol_rtms"),
    "age": ("age", "age_years", "âge"),
    "gender": ("gender", "sex", "genre"),
}

# Demographics are nice-to-have (the app's Patient card shows them) but never
# required: a download without them still loads.
_OPTIONAL_KEYS: tuple[str, ...] = ("age", "gender")


def _resolve_column(df: pd.DataFrame, explicit: str | None, key: str) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(f"column {explicit!r} not found in participants.tsv (have: {list(df.columns)})")
        return explicit
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in _COLUMN_CANDIDATES[key]:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # last resort: any column that startswith the semantic root (e.g. "bdi")
    root = key.split("_")[0]
    for low, orig in lower.items():
        if low.startswith(root):
            return orig
    raise KeyError(
        f"could not auto-resolve the {key!r} column in participants.tsv "
        f"(have: {list(df.columns)}); set TDBRAINConfig.col_{key} explicitly."
    )


def _read_participants(cfg: TDBRAINConfig) -> tuple[pd.DataFrame, dict[str, str | None]]:
    path = cfg.metadata_path or (cfg.root / "participants.tsv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Point TDBRAINConfig.root at your TDBRAIN copy "
            f"(the folder that contains participants.tsv), or set metadata_path. "
            f"See docs/tdbrain.md."
        )
    df = pd.read_csv(path, sep="\t", dtype=str, na_values=["n/a", "N/A", "", "NA"])

    # BDI/protocol are required for the responder task but irrelevant for diagnosis;
    # in diagnosis mode a missing column is tolerated (resolved to None) instead of raising.
    def _optional(explicit: str | None, key: str) -> str | None:
        if cfg.task == "diagnosis" and explicit is None:
            try:
                return _resolve_column(df, None, key)
            except KeyError:
                return None
        return _resolve_column(df, explicit, key)

    cols: dict[str, str | None] = {
        "id": _resolve_column(df, cfg.col_id, "id"),
        "indication": _resolve_column(df, cfg.col_indication, "indication"),
        "bdi_pre": _optional(cfg.col_bdi_pre, "bdi_pre"),
        "bdi_post": _optional(cfg.col_bdi_post, "bdi_post"),
        "protocol": _optional(cfg.col_protocol, "protocol"),
    }
    for key in _OPTIONAL_KEYS:
        try:
            cols[key] = _resolve_column(df, None, key)
        except KeyError:
            cols[key] = None
    return df, cols


# --------------------------------------------------------------------------- #
# EEG condition-file parsing
# --------------------------------------------------------------------------- #

# Real TDBRAIN ships EEG as BioSemi BDF; the synthetic fixture uses CSV. BDF is
# preferred when both are present for a subject.
_EEG_SUFFIXES: tuple[str, ...] = (".bdf", ".csv")


def _find_condition_file(cfg: TDBRAINConfig, subject_id: str) -> Path | None:
    """Locate a subject's eyes-open/closed recording.

    Tolerant to BIDS vs shorthand naming and to file format: ``.bdf`` (real
    TDBRAIN) is tried before ``.csv`` (synthetic fixture / derivative export).
    """
    cond = cfg.condition.upper()
    other = "EC" if cond == "EO" else "EO"
    for suffix in _EEG_SUFFIXES:
        patterns = [
            f"**/sub-{subject_id}*ses-{cfg.session}*{cond}*{suffix}",
            f"**/*{subject_id}*ses-{cfg.session}*{cond}*{suffix}",
            f"**/sub-{subject_id}*{cond}*{suffix}",
            f"**/*{subject_id}*{cond}*{suffix}",
        ]
        for pat in patterns:
            matches = sorted(cfg.root.rglob(pat))
            # Guard against EO/EC substring collisions (never let "EC" match an "EO" file).
            matches = [m for m in matches if f".{other}." not in m.name and f"_{other}" not in m.name]
            if matches:
                return matches[0]
    return None


def _select_channels(raw: pd.DataFrame, channels: tuple[str, ...]) -> np.ndarray | None:
    """Return (n_channels, n_samples) if `raw` has channel-named columns, else None."""
    lower = {str(c).lower().strip(): c for c in raw.columns}

    def find(ch: str) -> str | None:
        for name in (ch, *_CHANNEL_ALIASES.get(ch, ())):
            if name.lower() in lower:
                return lower[name.lower()]
        return None

    resolved = [find(ch) for ch in channels]
    if all(r is None for r in resolved):
        return None
    if any(r is None for r in resolved):
        missing = [ch for ch, r in zip(channels, resolved) if r is None]
        raise KeyError(f"channels missing from EEG CSV header: {missing}")
    return raw[resolved].to_numpy(dtype=np.float64).T


def _read_condition_csv(path: Path, channels: tuple[str, ...]) -> np.ndarray:
    """Load one condition file as (n_channels, n_samples) at SOURCE_FS.

    Primary path: a header row of channel names (the TDBRAIN derivative CSVs).
    Fallback: headerless numeric, oriented by whichever axis matches the montage.
    """
    raw = pd.read_csv(path)
    data = _select_channels(raw, channels)
    if data is not None:
        return data

    # Fallback: no channel-named header — re-read raw and infer orientation.
    mat = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
    n = len(channels)
    if mat.shape[0] >= n and mat.shape[0] <= mat.shape[1]:
        return mat[:n, :]                       # rows = channels
    if mat.shape[1] >= n and mat.shape[1] < mat.shape[0]:
        return mat[:, :n].T                     # cols = channels
    raise ValueError(
        f"cannot infer channel orientation of {path.name} (shape {mat.shape}, "
        f"expected one axis >= {n}); check the CSV layout against docs/tdbrain.md."
    )


def _read_condition_bdf(
    path: Path,
    channels: tuple[str, ...],
    notch_hz: float | None = None,
    bandpass_hz: tuple[float, float] | None = None,
) -> np.ndarray:
    """Load one BDF recording as (n_channels, n_samples) in microvolts at SOURCE_FS.

    Uses mne's built-in BDF reader (no extra dependency beyond ``mne`` itself).
    Channel names are matched case-insensitively, honouring the 10-20 aliases.
    Raw TDBRAIN EEG is unfiltered, so an optional power-line ``notch_hz`` (with
    harmonics) and ``bandpass_hz`` (high-pass, low-pass) are applied before the
    return. mne returns Volts; TDBRAIN units are microvolts, so we scale by 1e6.
    """
    import mne  # optional dep: only needed for real BDF data, not the CSV fixture

    raw = mne.io.read_raw_bdf(str(path), preload=True, verbose="ERROR")
    return _eeg_from_raw(raw, channels, path.name, notch_hz, bandpass_hz)


def _eeg_from_raw(
    raw,
    channels: tuple[str, ...],
    label: str,
    notch_hz: float | None,
    bandpass_hz: tuple[float, float] | None,
) -> np.ndarray:
    """Select, resample and filter the EEG montage from an open mne Raw."""
    present = {str(c).lower().strip(): c for c in raw.ch_names}

    def find(ch: str) -> str | None:
        for name in (ch, *_CHANNEL_ALIASES.get(ch, ())):
            if name.lower() in present:
                return present[name.lower()]
        return None

    resolved = [find(ch) for ch in channels]
    missing = [ch for ch, r in zip(channels, resolved) if r is None]
    if missing:
        raise KeyError(f"channels missing from BDF {label}: {missing}")

    data = raw.get_data(picks=resolved)                 # (n_channels, n_samples), Volts
    sfreq = float(raw.info["sfreq"])
    if abs(sfreq - SOURCE_FS) > 0.5:                    # keep the SOURCE_FS contract downstream
        data = _resample(data, sfreq, SOURCE_FS)
        sfreq = SOURCE_FS
    data = _filter_eeg(data, sfreq, notch_hz, bandpass_hz)
    return np.ascontiguousarray(data * 1e6, dtype=np.float64)   # Volts -> microvolts


def _filter_eeg(
    data: np.ndarray,
    fs: float,
    notch_hz: float | None,
    bandpass_hz: tuple[float, float] | None,
) -> np.ndarray:
    """Apply an optional power-line notch (+harmonics) and band-pass to (n_ch, n_samp).

    No-op when both are None. Uses mne's array filters (FIR), which operate on the
    last axis. Harmonics of ``notch_hz`` up to Nyquist are notched together.
    """
    if notch_hz is None and bandpass_hz is None:
        return data
    from mne.filter import filter_data, notch_filter

    if notch_hz is not None:
        nyq = fs / 2.0
        freqs = np.arange(notch_hz, nyq, notch_hz)
        if freqs.size:
            data = notch_filter(data, fs, freqs, verbose="ERROR")
    if bandpass_hz is not None:
        lo, hi = bandpass_hz
        data = filter_data(data, fs, lo, hi, verbose="ERROR")
    return data


def _read_ecg(path: Path, ecg_channel: str) -> tuple[np.ndarray, float]:
    """Read the autonomic lead from a recording, dispatching on file type.

    Mirrors :func:`_read_condition_file`. CSV is assumed to be at
    :data:`SOURCE_FS`, matching the assumption the EEG CSV path already makes.
    Prefer :func:`_read_recording` when the EEG is needed too — it avoids
    decoding the same BDF twice.
    """
    if path.suffix.lower() == ".bdf":
        return _read_ecg_bdf(path, ecg_channel)
    raw = pd.read_csv(path)
    lower = {str(c).lower().strip(): c for c in raw.columns}
    col = lower.get(ecg_channel.lower().strip())
    if col is None:
        return np.zeros(0, dtype=np.float64), SOURCE_FS
    return raw[col].to_numpy(dtype=np.float64), SOURCE_FS


def _extract_ecg(raw, ecg_channel: str) -> tuple[np.ndarray, float]:
    """Pull the autonomic lead out of an open mne Raw as ``(samples, fs)`` in uV.

    Left **unfiltered**: the EEG notch and the downsample to ``target_fs`` both
    blunt R-peak timing, and RR precision is the whole point, so detection runs at
    the native 500 Hz. Returns an empty array when the montage has no such channel.
    """
    present = {str(c).lower().strip(): c for c in raw.ch_names}
    name = present.get(ecg_channel.lower().strip())
    if name is None:
        return np.zeros(0, dtype=np.float64), float(raw.info["sfreq"])
    data = raw.get_data(picks=[name])[0] * 1e6          # Volts -> microvolts
    return np.ascontiguousarray(data, dtype=np.float64), float(raw.info["sfreq"])


def _read_ecg_bdf(path: Path, ecg_channel: str) -> tuple[np.ndarray, float]:
    """Standalone ECG read (opens the file itself). See :func:`_extract_ecg`."""
    import mne

    raw = mne.io.read_raw_bdf(str(path), preload=True, verbose="ERROR")
    return _extract_ecg(raw, ecg_channel)


def _read_recording(
    path: Path,
    channels: tuple[str, ...],
    ecg_channel: str | None,
    notch_hz: float | None,
    bandpass_hz: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Read EEG montage **and** autonomic lead from one recording.

    Returns ``(eeg, ecg_raw, ecg_fs)``. Decoding a 120 s 33-channel BDF is the
    dominant cost of loading the cohort, so both modalities come out of a single
    open — reading the file once per modality doubled the whole load.
    """
    if path.suffix.lower() != ".bdf":
        eeg = _read_condition_csv(path, channels)
        if ecg_channel is None:
            return eeg, np.zeros(0, dtype=np.float64), SOURCE_FS
        ecg_raw, ecg_fs = _read_ecg(path, ecg_channel)
        return eeg, ecg_raw, ecg_fs

    import mne

    raw = mne.io.read_raw_bdf(str(path), preload=True, verbose="ERROR")
    ecg_raw, ecg_fs = (
        _extract_ecg(raw, ecg_channel) if ecg_channel
        else (np.zeros(0, dtype=np.float64), float(raw.info["sfreq"]))
    )
    eeg = _eeg_from_raw(raw, channels, path.name, notch_hz, bandpass_hz)
    return eeg, ecg_raw, ecg_fs


def detect_rr_intervals(
    ecg: np.ndarray,
    fs: float,
    bandpass_hz: tuple[float, float] | None = (5.0, 20.0),
    min_bpm: float = 30.0,
    max_bpm: float = 200.0,
) -> np.ndarray:
    """R-peak detection on a raw ECG lead -> RR intervals in **seconds**.

    Deliberately simple (band-pass -> rectify -> peak-pick -> physiological
    gating), because TDBRAIN's ``Erbs`` lead is clean enough that a full
    Pan-Tompkins buys nothing. Two rejection stages keep artefacts out of HRV:

    1. *Physiological* — intervals outside ``[60/max_bpm, 60/min_bpm]`` are dropped
       (a missed beat doubles an interval; a spurious one halves it).
    2. *Robust outlier* — intervals deviating from the median by more than 40% are
       dropped. Without this a handful of missed beats inflate SDNN by ~10x, which
       is exactly what a naive detector does on the noisier recordings.

    Returns an empty array when the lead is flat or unusable; callers treat that
    as "no autonomic data for this patient" rather than failing the whole load.
    """
    x = np.asarray(ecg, dtype=np.float64)
    if x.size < int(2 * fs) or not np.isfinite(x).all() or np.std(x) == 0:
        return np.zeros(0, dtype=np.float64)

    from scipy.signal import butter, filtfilt, find_peaks

    if bandpass_hz is not None:
        lo, hi = bandpass_hz
        nyq = fs / 2.0
        b, a = butter(3, [lo / nyq, min(hi / nyq, 0.99)], btype="band")
        x = filtfilt(b, a, x)

    # Pan-Tompkins energy envelope: derivative -> square -> moving-window integrate.
    # Squaring the slope suppresses the T wave (low slope, comparable amplitude),
    # which a plain |x| threshold happily mistakes for a second R peak — the failure
    # that produces alternating short/long intervals and a 4x-inflated RMSSD.
    deriv = np.diff(x, prepend=x[0])
    energy = deriv * deriv
    win = max(1, int(round(0.15 * fs)))                 # ~QRS width
    env = np.convolve(energy, np.ones(win) / win, mode="same")

    # Percentile threshold, not mean+k*std: a handful of saturation spikes drag the
    # std far above the true QRS amplitude and starve the detector of peaks.
    thresh = 0.5 * float(np.percentile(env, 98))
    if thresh <= 0:
        return np.zeros(0, dtype=np.float64)
    peaks, _ = find_peaks(env, height=thresh, distance=int(round(60.0 / max_bpm * fs)))
    if peaks.size < 3:
        return np.zeros(0, dtype=np.float64)

    rr = np.diff(peaks) / float(fs)
    rr = rr[(rr >= 60.0 / max_bpm) & (rr <= 60.0 / min_bpm)]
    if rr.size < 3:
        return np.zeros(0, dtype=np.float64)
    return _malik_filter(rr)


def _malik_filter(rr: np.ndarray, tol: float = 0.2) -> np.ndarray:
    """Standard HRV artefact correction: drop beats deviating >``tol`` from context.

    Each interval is compared against the running median of those already accepted
    (seeded with the global median). A missed beat yields a ~2x interval and a
    spurious one a ~0.5x interval; both fail the test, whereas genuine respiratory
    sinus arrhythmia (typically <15%) passes. Applied *after* the physiological
    gate because that gate only catches absolute outliers, not local jumps.
    """
    if rr.size < 3:
        return rr
    kept: list[float] = []
    ref = float(np.median(rr))
    for v in rr:
        if abs(v - ref) <= tol * ref:
            kept.append(float(v))
            ref = float(np.median(kept[-8:]))            # local, adapts to HR drift
    return np.asarray(kept, dtype=np.float64) if len(kept) >= 3 else np.zeros(0, dtype=np.float64)


def _tachogram(rr: np.ndarray, n_rr: int) -> np.ndarray:
    """Fit a variable-length RR series to the fixed ``(n_rr,)`` tensor slot.

    Longer series are truncated (TDBRAIN's 120 s gives ~140 beats, so ``n_rr=64``
    is a plain head-slice). Shorter ones are **tiled**, not zero-padded: padding
    with zeros or a constant would deflate SDNN/RMSSD toward zero and fabricate a
    low-variability patient. An all-empty series yields zeros, which
    :func:`~src.preprocessing.features.hrv_features` maps to neutral values.
    """
    if rr.size == 0:
        return np.zeros(n_rr, dtype=np.float32)
    if rr.size >= n_rr:
        return rr[:n_rr].astype(np.float32)
    reps = int(np.ceil(n_rr / rr.size))
    return np.tile(rr, reps)[:n_rr].astype(np.float32)


def _read_condition_file(
    path: Path,
    channels: tuple[str, ...],
    notch_hz: float | None = None,
    bandpass_hz: tuple[float, float] | None = None,
) -> np.ndarray:
    """Read a recording as (n_channels, n_samples) at SOURCE_FS, dispatching on type:
    BDF (real TDBRAIN) via mne with optional filtering, CSV (synthetic fixture /
    derivative export, assumed clean) via pandas.
    """
    if path.suffix.lower() == ".bdf":
        return _read_condition_bdf(path, channels, notch_hz, bandpass_hz)
    return _read_condition_csv(path, channels)


def _resample(x: np.ndarray, src_fs: float, dst_fs: float) -> np.ndarray:
    if src_fs == dst_fs:
        return x
    g = gcd(int(round(src_fs)), int(round(dst_fs)))
    up = int(round(dst_fs)) // g
    down = int(round(src_fs)) // g
    return resample_poly(x, up, down, axis=-1)


def _epoch(data: np.ndarray, cfg: TDBRAINConfig, subject_id: str) -> np.ndarray:
    """(n_channels, n_samples) -> (n_epochs, n_channels, window) at target_fs."""
    win = int(round(cfg.epoch_seconds * cfg.target_fs))
    usable = win * cfg.n_epochs
    n_samples = data.shape[1]
    if n_samples < usable:
        warnings.warn(
            f"{subject_id}: recording has {n_samples} samples < {usable} needed "
            f"({cfg.n_epochs}x{win}); edge-padding to fit.",
            stacklevel=2,
        )
        data = np.pad(data, ((0, 0), (0, usable - n_samples)), mode="edge")
    block = data[:, :usable]                                    # (n_channels, usable)
    if cfg.snapshot:
        return block[None, :, :]                               # (1, n_channels, usable)
    n_ch = block.shape[0]
    return block.reshape(n_ch, cfg.n_epochs, win).transpose(1, 0, 2)  # (n_epochs, n_channels, win)


# --------------------------------------------------------------------------- #
# Public loader
# --------------------------------------------------------------------------- #

def load_tdbrain(cfg: TDBRAINConfig) -> LoadedDataset:
    """Load the TDBRAIN rTMS-in-MDD cohort into a :class:`LoadedDataset`.

    ``signals`` is the single ``representative_channel`` (for the app's raw trace);
    ``signals_mc`` holds the full montage ``(n_patients, n_epochs, n_channels, window)``
    for modelling via :func:`tdbrain_features`.
    """
    df, cols = _read_participants(cfg)
    if cfg.representative_channel not in cfg.channels:
        raise ValueError(f"representative_channel {cfg.representative_channel!r} not in channels")
    rep_idx = cfg.channels.index(cfg.representative_channel)

    win = int(round(cfg.epoch_seconds * cfg.target_fs))
    window = win * cfg.n_epochs if cfg.snapshot else win

    mc_list: list[np.ndarray] = []
    rr_list: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict] = []
    skipped = _Skipped()
    n_no_ecg = 0

    for _, r in df.iterrows():
        indication = str(r[cols["indication"]] or "")
        is_positive = cfg.indication.lower() in indication.lower()

        # --- per-task eligibility + label + metadata ------------------------ #
        if cfg.task == "diagnosis":
            if is_positive:
                label = 1
            elif indication.strip().upper() == cfg.control_indication.upper():
                label = 0
            else:
                skipped.add("indication")
                continue
            meta: dict = {"indication": indication, "label": label}
        else:  # "response"
            if not is_positive:
                skipped.add("indication")
                continue
            try:
                protocol = int(float(r[cols["protocol"]]))
            except (ValueError, TypeError):
                skipped.add("no-protocol")
                continue
            if protocol not in cfg.protocols:
                skipped.add("protocol-filtered")
                continue
            bdi_pre = pd.to_numeric(r[cols["bdi_pre"]], errors="coerce")
            bdi_post = pd.to_numeric(r[cols["bdi_post"]], errors="coerce")
            if not np.isfinite(bdi_pre) or not np.isfinite(bdi_post) or bdi_pre <= 0:
                skipped.add("no-bdi")
                continue
            pct = float((bdi_pre - bdi_post) / bdi_pre)
            label = int(pct >= cfg.responder_threshold)
            meta = {
                "protocol": protocol,
                "bdi_pre": float(bdi_pre),
                "bdi_post": float(bdi_post),
                "delta_bdi": float(bdi_pre - bdi_post),
                "pct_reduction": pct,
                "responder": label,
            }

        subject_id = str(r[cols["id"]]).replace("sub-", "")
        rec_path = _find_condition_file(cfg, subject_id)
        if rec_path is None:
            skipped.add("no-eeg-file")
            continue
        try:
            data, ecg_raw, ecg_fs = _read_recording(
                rec_path, cfg.channels, cfg.ecg_channel, cfg.notch_hz, cfg.bandpass_hz
            )
            data = _resample(data, SOURCE_FS, cfg.target_fs)
            epochs = _epoch(data, cfg, subject_id).astype(np.float32)  # (n_seq, n_ch, window)
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            warnings.warn(f"{subject_id}: skipped ({exc})", stacklevel=2)
            skipped.add("unreadable-eeg")
            continue

        # Autonomic lead, from the same read. A patient without a usable ECG is kept
        # (EEG is the primary modality) and gets a zero tachogram, which
        # hrv_features maps to neutral values — losing 26 EEG channels to salvage
        # 5 HRV numbers would be a bad trade.
        rr = np.zeros(0, dtype=np.float64)
        if cfg.ecg_channel and ecg_raw.size:
            try:
                rr = detect_rr_intervals(ecg_raw, ecg_fs, cfg.ecg_bandpass_hz)
            except (ValueError, RuntimeError) as exc:
                warnings.warn(f"{subject_id}: ECG unusable ({exc})", stacklevel=2)
        if rr.size == 0 and cfg.ecg_channel:
            n_no_ecg += 1
        rr_list.append(_tachogram(rr, cfg.n_rr))

        mc_list.append(epochs)
        labels.append(label)
        rows.append({"patient_id": subject_id, **meta, **_demographics(r, cols)})

    if skipped.reasons:
        warnings.warn(f"TDBRAIN: excluded patients by reason: {skipped.reasons}", stacklevel=2)
    if not mc_list:
        raise ValueError(
            "no eligible TDBRAIN patients loaded — check indication/protocol filters, "
            "BDI columns, and EEG file discovery against docs/tdbrain.md."
        )

    signals_mc = np.stack(mc_list, axis=0)                 # (n_patients, n_seq, n_ch, window)
    signals = signals_mc[:, :, rep_idx, :]                 # (n_patients, n_seq, window)
    metadata = pd.DataFrame(rows)

    # HRV is a **patient-level** trait here, not a per-epoch one: an 8 s epoch holds
    # ~9 beats, far too few for SDNN or LF/HF (hrv_features needs >=16). So the
    # tachogram is measured over the whole recording and repeated along the epoch
    # axis — constant across the sequence, exactly like the BDI scores. See
    # tdbrain_features() for why that block must skip per-epoch z-scoring.
    # No usable lead anywhere (CSV fixture, or a montage without Erbs) means the
    # modality is absent, not empty: report None so callers fail loudly on
    # modalities=("ecg",) instead of silently training on a block of zeros.
    ecg = None
    if cfg.ecg_channel and rr_list and n_no_ecg < len(rr_list):
        n_seq = signals_mc.shape[1]
        ecg = np.repeat(np.stack(rr_list, axis=0)[:, None, :], n_seq, axis=1)
        if n_no_ecg:
            warnings.warn(
                f"TDBRAIN: {n_no_ecg}/{len(rr_list)} patients have no usable ECG "
                f"(zero tachogram -> neutral HRV features).",
                stacklevel=2,
            )

    return LoadedDataset(
        signals=np.ascontiguousarray(signals),
        labels=np.asarray(labels, dtype=np.int8),
        fs=cfg.target_fs,
        window=window,
        metadata=metadata,
        erp=None,
        ecg=ecg,
        channels=list(cfg.channels),
        signals_mc=signals_mc,
    )


def montage_feature_names(channels) -> tuple[str, ...]:
    """Feature-vector column names for a montage: channel-major, band-minor."""
    return tuple(f"{ch}_power_{b}" for ch in channels for b in BANDS)


def montage_band_powers(mc: np.ndarray, fs: float) -> np.ndarray:
    """``(n_epochs, n_channels, window)`` -> ``(n_epochs, n_channels * n_bands)``.

    Relative band powers, channel-major/band-minor — the same ordering as
    :func:`montage_feature_names`. Shared by :func:`tdbrain_features` (training,
    whole cohort) and by the Streamlit app (inference, one patient), so a model
    can never be fed a differently-ordered vector than it was trained on.
    """
    if mc.ndim != 3:
        raise ValueError(f"expected (n_epochs, n_channels, window); got {mc.shape}")
    n_seq, n_ch, _ = mc.shape
    band_names = list(BANDS)
    out = np.empty((n_seq, n_ch * len(band_names)), dtype=np.float32)
    for s in range(n_seq):
        vec: list[float] = []
        for c in range(n_ch):
            feats = basic_features(mc[s, c], fs)
            vec.extend(feats[f"power_{b}"] for b in band_names)
        out[s] = vec
    return out


def zscore_epochs(x: np.ndarray) -> np.ndarray:
    """Per-patient z-score across the epoch axis of ``(..., n_epochs, n_features)``.

    Uses only the patient's own epochs, so it is identical whether applied to a
    whole cohort at training time or to a single patient at inference time.
    """
    mean = x.mean(axis=-2, keepdims=True)
    std = x.std(axis=-2, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def tdbrain_hrv_block(dataset: LoadedDataset) -> tuple[np.ndarray, tuple[str, ...]]:
    """``(n_patients, n_epochs, 5)`` time-domain HRV features from the ECG tachogram.

    Reuses the simulated track's :func:`~src.preprocessing.features.hrv_features`
    so both cohorts report the same autonomic metrics under the same names.
    """
    from ..preprocessing.features import hrv_features
    from ..preprocessing.pipeline import HRV_FEATURE_NAMES

    ecg = dataset.ecg
    if ecg is None:
        raise ValueError("modality 'ecg' requested but dataset.ecg is None")
    n_p, n_seq, _ = ecg.shape
    out = np.empty((n_p, n_seq, len(HRV_FEATURE_NAMES)), dtype=np.float32)
    for p in range(n_p):
        for s in range(n_seq):
            feats = hrv_features(ecg[p, s])
            out[p, s] = [feats[name] for name in HRV_FEATURE_NAMES]
    return out, tuple(HRV_FEATURE_NAMES)


def tdbrain_features(
    dataset: LoadedDataset,
    per_patient_zscore: bool = True,
    modalities: tuple[str, ...] = ("eeg",),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Per-epoch features for ``cross_validate``, over the requested modalities.

    Thin wrapper kept for the existing call sites. The implementation now lives in
    :func:`src.data.modalities.build_features`, which serves the real and the
    simulated cohort through one code path — the four-model comparison is only
    like-for-like if both are assembled by the same code. ``modalities`` also
    accepts ``"rtms"`` (the clinical block) there.
    """
    from .modalities import build_features

    return build_features(
        dataset, modalities=modalities, per_patient_zscore=per_patient_zscore
    )


# --------------------------------------------------------------------------- #
# Synthetic fixture — a tiny TDBRAIN-shaped tree for tests / dry runs.
# --------------------------------------------------------------------------- #

def _write_synthetic_bdf(path: Path, data_uv: np.ndarray, channels: tuple[str, ...]) -> None:
    """Write (n_channels, n_samples) microvolt data as a BDF file (test fixture only).

    Requires ``mne`` + ``edfio`` (mne's BDF exporter). Only used to exercise the real
    BDF read path against a synthetic tree; never touches patient data.
    """
    import mne  # optional: BDF writing needs mne + edfio, reading needs only mne

    info = mne.create_info(list(channels), SOURCE_FS, ch_types="eeg")
    raw = mne.io.RawArray(np.asarray(data_uv, dtype=np.float64) * 1e-6, info, verbose="ERROR")
    raw.export(str(path), fmt="bdf", overwrite=True, verbose="ERROR")


def _synthetic_qrs(n_samples: int, fs: float, bpm: float, rng) -> np.ndarray:
    """A crude but detectable ECG: sharp R spikes plus a broad T wave.

    The T wave matters — it is the thing a naive ``|x| > k*sigma`` detector
    mistakes for a second beat, so including it lets the tests exercise the
    rejection logic in :func:`detect_rr_intervals` rather than a trivially clean
    signal. Amplitudes are in microvolts, R ~1 mV as at Erb's point.
    """
    x = np.zeros(n_samples, dtype=np.float64)
    rr = 60.0 / bpm
    t_rel = np.arange(n_samples) / fs

    def _gaussian(centre_idx: int, sigma_s: float, amp: float) -> None:
        sigma = max(1.0, sigma_s * fs)
        half = int(round(4 * sigma))
        lo, hi = max(0, centre_idx - half), min(n_samples, centre_idx + half)
        if hi > lo:
            k = np.arange(lo, hi) - centre_idx
            x[lo:hi] += amp * np.exp(-0.5 * (k / sigma) ** 2)

    beat = 0.0
    while beat < n_samples / fs:
        idx = int(round(beat * fs))
        if 0 <= idx < n_samples:
            # QRS: narrow (sigma ~10 ms) and tall -> steep slope. T: broad
            # (sigma ~50 ms) and short -> shallow slope. Pan-Tompkins separates the
            # two by slope, so the ratio here is what makes the T wave rejectable.
            _gaussian(idx, 0.010, 1000.0)
            _gaussian(idx + int(round(0.30 * fs)), 0.050, 200.0)
        beat += rr * float(1.0 + 0.04 * rng.standard_normal())  # mild sinus arrhythmia
    return x + 5.0 * rng.standard_normal(n_samples) + 20.0 * np.sin(2 * np.pi * 0.3 * t_rel)


def make_synthetic_tdbrain(
    root: Path,
    n_patients: int = 12,
    duration_seconds: float = 10.0,
    seed: int = 0,
    include_noise_rows: bool = True,
    fmt: str = "csv",
    with_ecg: bool = False,
) -> Path:
    """Write a miniature TDBRAIN-format tree (participants.tsv + EO/EC recordings).

    Responders (>=50% BDI reduction) are given elevated eyes-open alpha power so the
    full load -> features -> LSTM path has something learnable. Not medical data —
    purely for exercising the parser and pipeline. Returns ``root``.

    ``fmt`` selects the recording format: ``"csv"`` (default, dependency-free) or
    ``"bdf"`` (real TDBRAIN's format; needs ``mne`` + ``edfio``). ``with_ecg`` adds
    an ``Erbs`` lead so the autonomic path can be tested; responders are given a
    slower heart rate, mirroring the elevated alpha, so HRV is learnable too.
    """
    if fmt not in ("csv", "bdf"):
        raise ValueError(f"fmt must be 'csv' or 'bdf', got {fmt!r}")
    root = Path(root)
    (root).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_samples = int(round(duration_seconds * SOURCE_FS))
    t = np.arange(n_samples) / SOURCE_FS
    channels = TDBRAIN_CHANNELS_26

    participant_rows: list[dict] = []
    for i in range(n_patients):
        subject_id = f"{10000000 + i}"
        responder = i % 2 == 0                      # 6/12 responders, alternating
        protocol = 1 if i < n_patients // 2 else 2  # protocol independent of response
        bdi_pre = float(rng.uniform(20, 35))
        bdi_post = bdi_pre * (0.25 if responder else 0.9)   # 75% vs 10% reduction
        participant_rows.append({
            "participants_ID": f"sub-{subject_id}",
            "indication": "MDD",
            "age": int(rng.integers(25, 65)),
            "gender": int(rng.integers(0, 2)),
            "rTMS_protocol": protocol,
            "BDI_pre": round(bdi_pre, 1),
            "BDI_post": round(bdi_post, 1),
        })
        alpha_amp = 2.2 if responder else 0.8
        out_channels = (*channels, "Erbs") if with_ecg else channels
        for cond in ("EO", "EC"):
            eo = cond == "EO"
            sig = np.empty((n_samples, len(out_channels)), dtype=np.float32)
            for c in range(len(channels)):
                theta = 1.0 * np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 2 * np.pi))
                a_amp = alpha_amp if eo else 1.0     # discriminative signal lives in EO
                alpha = a_amp * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))
                beta = 0.5 * np.sin(2 * np.pi * 20.0 * t + rng.uniform(0, 2 * np.pi))
                noise = 0.6 * rng.standard_normal(n_samples)
                sig[:, c] = (theta + alpha + beta + noise).astype(np.float32)
            if with_ecg:
                sig[:, -1] = _synthetic_qrs(
                    n_samples, SOURCE_FS, 58.0 if responder else 78.0, rng
                ).astype(np.float32)
            ses_dir = root / "derivatives" / f"sub-{subject_id}" / "ses-1" / "eeg"
            ses_dir.mkdir(parents=True, exist_ok=True)
            stem = f"sub-{subject_id}_ses-1_task-rest{cond}_eeg"
            if fmt == "bdf":
                _write_synthetic_bdf(ses_dir / f"{stem}.bdf", sig.T, out_channels)
            else:
                pd.DataFrame(sig, columns=list(out_channels)).to_csv(
                    ses_dir / f"{stem}.csv", index=False
                )

    if include_noise_rows:
        # A protocol-3 MDD patient and a non-MDD patient — both must be filtered out.
        participant_rows.append({
            "participants_ID": "sub-19000001", "indication": "MDD", "age": 50, "gender": 1,
            "rTMS_protocol": 3, "BDI_pre": 30.0, "BDI_post": 10.0,
        })
        participant_rows.append({
            "participants_ID": "sub-19000002", "indication": "ADHD", "age": 30, "gender": 0,
            "rTMS_protocol": 1, "BDI_pre": "n/a", "BDI_post": "n/a",
        })

    pd.DataFrame(participant_rows).to_csv(root / "participants.tsv", sep="\t", index=False)
    return root


if __name__ == "__main__":
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(description="Dry-run the TDBRAIN loader on a synthetic fixture.")
    ap.add_argument("--root", type=Path, default=None, help="TDBRAIN root (default: temp synthetic tree)")
    ap.add_argument("--fmt", choices=("csv", "bdf"), default="csv",
                    help="synthetic recording format when --root is not given (default: csv)")
    args = ap.parse_args()

    root = args.root
    if root is None:
        root = Path(tempfile.mkdtemp()) / "tdbrain_synth"
        make_synthetic_tdbrain(root, fmt=args.fmt)
        print(f"Wrote synthetic fixture ({args.fmt}) to {root}")

    ds = load_tdbrain(TDBRAINConfig(root=root, n_epochs=4, epoch_seconds=1.0))
    x, y, groups, names = tdbrain_features(ds)
    print(f"patients={ds.signals.shape[0]}  epochs={ds.signals.shape[1]}  window={ds.window}")
    print(f"montage={None if ds.signals_mc is None else ds.signals_mc.shape}  features={x.shape} ({len(names)} names)")
    print(f"responders={int(y.sum())}/{len(y)}  protocols={sorted(ds.metadata['protocol'].unique())}")
