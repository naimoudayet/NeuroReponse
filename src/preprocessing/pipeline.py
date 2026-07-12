"""End-to-end preprocessing for the (n_patients, n_sessions, window) tensor.

Two output modes:
  - "raw":     bandpass + per-patient z-score, shape (n_patients, n_sessions, window)
  - "features": per-session feature vector (mean/std/rms + 5 band powers),
                shape (n_patients, n_sessions, n_features)

Normalization is computed *per patient* (across that patient's sessions only),
never across patients — prevents target leakage in patient-wise CV.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .features import BANDS, basic_features, erp_features, hrv_features
from .filters import bandpass

Mode = Literal["raw", "features"]

FEATURE_NAMES: tuple[str, ...] = (
    "mean",
    "std",
    "rms",
    *(f"power_{name}" for name in BANDS),
)

ERP_FEATURE_NAMES: tuple[str, ...] = (
    "erp_n100_amp", "erp_n100_lat", "erp_p300_amp", "erp_p300_lat", "erp_auc",
)
HRV_FEATURE_NAMES: tuple[str, ...] = (
    "hrv_hr_mean", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_lf_hf",
)

# Feature block contributed by each physiological modality (see RES0_AR1 / NPDT).
MODALITY_FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "eeg": FEATURE_NAMES,
    "erp": ERP_FEATURE_NAMES,
    "ecg": HRV_FEATURE_NAMES,
}


@dataclass
class PipelineConfig:
    fs: float = 256.0
    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 40.0
    mode: Mode = "features"
    apply_bandpass: bool = True
    per_patient_zscore: bool = True


@dataclass
class PreprocessedDataset:
    x: np.ndarray
    feature_names: tuple[str, ...] | None
    config: PipelineConfig | MultimodalConfig
    fitted_stats: dict = field(default_factory=dict)


def _bandpass_3d(signals: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    flat = signals.reshape(-1, signals.shape[-1])
    filtered = bandpass(flat, cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.fs)
    return filtered.reshape(signals.shape).astype(np.float32)


def _per_patient_zscore(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # signals: (n_patients, n_sessions, window) — flatten sessions+window per patient
    flat = signals.reshape(signals.shape[0], -1)
    mean = flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    normalized = (flat - mean) / std
    return (
        normalized.reshape(signals.shape).astype(np.float32),
        mean.squeeze(1),
        std.squeeze(1),
    )


def _features_per_session(signals: np.ndarray, fs: float) -> np.ndarray:
    n_patients, n_sessions, _ = signals.shape
    out = np.empty((n_patients, n_sessions, len(FEATURE_NAMES)), dtype=np.float32)
    for p in range(n_patients):
        for s in range(n_sessions):
            feats = basic_features(signals[p, s], fs)
            out[p, s] = [feats[name] for name in FEATURE_NAMES]
    return out


def _zscore_features_per_patient(features: np.ndarray) -> np.ndarray:
    # (n_patients, n_sessions, n_features) — normalize each feature within each patient
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return ((features - mean) / std).astype(np.float32)


def preprocess(signals: np.ndarray, cfg: PipelineConfig | None = None) -> PreprocessedDataset:
    cfg = cfg or PipelineConfig()
    if signals.ndim != 3:
        raise ValueError(f"signals must be 3-D (patients, sessions, window); got shape {signals.shape}")

    work = _bandpass_3d(signals, cfg) if cfg.apply_bandpass else signals.astype(np.float32)

    stats: dict = {}
    if cfg.mode == "raw":
        if cfg.per_patient_zscore:
            work, stats["mean"], stats["std"] = _per_patient_zscore(work)
        return PreprocessedDataset(x=work, feature_names=None, config=cfg, fitted_stats=stats)

    feats = _features_per_session(work, cfg.fs)
    if cfg.per_patient_zscore:
        feats = _zscore_features_per_patient(feats)
    return PreprocessedDataset(x=feats, feature_names=FEATURE_NAMES, config=cfg, fitted_stats=stats)


# --------------------------------------------------------------------------- #
# Multimodal preprocessing (EEG + ERP + ECG) — enables modality ablation.
# --------------------------------------------------------------------------- #

@dataclass
class MultimodalConfig:
    fs: float = 256.0
    modalities: tuple[str, ...] = ("eeg", "erp", "ecg")
    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 40.0
    apply_bandpass: bool = True          # applies to EEG only
    per_patient_zscore: bool = True


def _erp_features_per_session(erp: np.ndarray, fs: float) -> np.ndarray:
    n_patients, n_sessions, _ = erp.shape
    out = np.empty((n_patients, n_sessions, len(ERP_FEATURE_NAMES)), dtype=np.float32)
    for p in range(n_patients):
        for s in range(n_sessions):
            feats = erp_features(erp[p, s], fs)
            out[p, s] = [feats[name] for name in ERP_FEATURE_NAMES]
    return out


def _hrv_features_per_session(ecg: np.ndarray) -> np.ndarray:
    n_patients, n_sessions, _ = ecg.shape
    out = np.empty((n_patients, n_sessions, len(HRV_FEATURE_NAMES)), dtype=np.float32)
    for p in range(n_patients):
        for s in range(n_sessions):
            feats = hrv_features(ecg[p, s])
            out[p, s] = [feats[name] for name in HRV_FEATURE_NAMES]
    return out


def preprocess_multimodal(
    eeg: np.ndarray,
    erp: np.ndarray | None = None,
    ecg: np.ndarray | None = None,
    cfg: MultimodalConfig | None = None,
) -> PreprocessedDataset:
    """Per-session feature vectors fused across the selected modalities.

    Output shape: (n_patients, n_sessions, sum of selected modality feature counts).
    Normalization is per-patient per-feature — same leakage guarantee as `preprocess`.
    Selecting a subset of `cfg.modalities` is exactly the ablation knob the reviewers
    of the NPDT design will ask for (EEG vs EEG+ERP vs EEG+ECG vs all).
    """
    cfg = cfg or MultimodalConfig()
    if eeg.ndim != 3:
        raise ValueError(f"eeg must be 3-D (patients, sessions, window); got {eeg.shape}")
    unknown = set(cfg.modalities) - set(MODALITY_FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown modalities {sorted(unknown)}; choose from {list(MODALITY_FEATURE_NAMES)}")
    if not cfg.modalities:
        raise ValueError("at least one modality must be selected")

    blocks: list[np.ndarray] = []
    names: list[str] = []
    for m in cfg.modalities:
        if m == "eeg":
            work = _bandpass_3d(eeg, PipelineConfig(
                fs=cfg.fs, bandpass_low_hz=cfg.bandpass_low_hz, bandpass_high_hz=cfg.bandpass_high_hz,
            )) if cfg.apply_bandpass else eeg.astype(np.float32)
            blocks.append(_features_per_session(work, cfg.fs))
        elif m == "erp":
            if erp is None:
                raise ValueError("modality 'erp' requested but erp array is None")
            blocks.append(_erp_features_per_session(erp, cfg.fs))
        elif m == "ecg":
            if ecg is None:
                raise ValueError("modality 'ecg' requested but ecg array is None")
            blocks.append(_hrv_features_per_session(ecg))
        names.extend(MODALITY_FEATURE_NAMES[m])

    feats = np.concatenate(blocks, axis=-1).astype(np.float32)
    if cfg.per_patient_zscore:
        feats = _zscore_features_per_patient(feats)
    return PreprocessedDataset(x=feats, feature_names=tuple(names), config=cfg, fitted_stats={})
