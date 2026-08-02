"""Assemble model inputs from a :class:`LoadedDataset`, one block per modality.

One function serves both cohorts. That is the point: the four-model comparison is
only meaningful if "EEG + ECG on simulated data" and "EEG + ECG on TDBRAIN" are
built by the *same* code from the *same* contract. Two parallel implementations
would drift, and the comparison would quietly stop being like-for-like.

Three blocks are available:

``rtms``
    The **clinical** block: stimulation protocol plus the covariates known before
    treatment (age, gender, baseline BDI-II). Only the protocol *number* is taken
    from the stimulation parameters — frequency and site are deterministic
    functions of it (perfectly collinear), and intensity / train count / interval
    are not published by TDBRAIN at all. Including them would add columns of
    zeros or duplicated information, not signal.

``eeg``
    ``n_channels x n_bands`` relative band powers (130 for the 26-channel montage).

``ecg``
    Five time-domain HRV features from the RR tachogram.

**Two blocks are patient-level, and that governs normalisation.** The clinical and
HRV blocks are constant along the epoch axis by construction, so per-patient
z-scoring across epochs would divide by a zero standard deviation and collapse
them to constant zero — deleting the modality while every shape assertion still
passes. Only the EEG block is z-scored. ``tests/test_modalities.py`` pins this.

**Block order is canonical, not argument order.** Blocks are always concatenated
as ``rtms, eeg, ecg`` regardless of how ``modalities`` is spelled, so a checkpoint
can never be fed a permuted feature vector — the same class of bug that selecting
EEG channels by name (rather than row order) prevents elsewhere.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..preprocessing.features import BANDS
from .loader import LoadedDataset

MODALITY_ORDER: tuple[str, ...] = ("rtms", "eeg", "ecg")

# Clinical block. `rtms_protocol` is the stimulation arm; the rest are the
# pre-treatment covariates a clinician has before deciding to treat.
RTMS_FEATURE_NAMES: tuple[str, ...] = (
    "rtms_protocol", "age", "gender", "bdi_pre",
)

# Metadata columns the clinical block needs.
_RTMS_COLUMNS: dict[str, str] = {
    "rtms_protocol": "protocol",
    "age": "age",
    "gender": "gender",
    "bdi_pre": "bdi_pre",
}


class ModalityError(ValueError):
    """A requested modality cannot be built from this dataset."""


def _impute(values: np.ndarray, name: str) -> np.ndarray:
    """Replace missing entries with the cohort median, and say so.

    TDBRAIN's participants table is not always complete. Dropping a patient to
    save one covariate would cost 26 EEG channels; imputing keeps them. The
    median is computed over the whole cohort rather than per fold — a mild
    optimism, warned about rather than hidden, and in practice these columns are
    ~100% filled.
    """
    finite = np.isfinite(values)
    if finite.all():
        return values
    n_missing = int((~finite).sum())
    fill = float(np.median(values[finite])) if finite.any() else 0.0
    warnings.warn(
        f"clinical feature {name!r}: {n_missing} missing value(s) imputed "
        f"with the cohort median ({fill:.2f})",
        stacklevel=3,
    )
    out = values.copy()
    out[~finite] = fill
    return out


def rtms_block(dataset: LoadedDataset, n_epochs: int) -> tuple[np.ndarray, tuple[str, ...]]:
    """``(n_patients, n_epochs, 4)`` clinical features, repeated along the epochs.

    Repeated rather than varying because these are properties of the patient and
    the treatment course, not of the recording window — exactly like the HRV
    block. See the module docstring on why that forbids z-scoring them.
    """
    md = dataset.metadata
    missing = [c for c in _RTMS_COLUMNS.values() if c not in md.columns]
    if missing:
        raise ModalityError(
            f"modality 'rtms' needs metadata columns {missing}; the dataset "
            f"provides {sorted(md.columns)}"
        )

    cols = []
    for feat in RTMS_FEATURE_NAMES:
        raw = md[_RTMS_COLUMNS[feat]].to_numpy(dtype=np.float64, na_value=np.nan)
        cols.append(_impute(raw, feat))
    per_patient = np.stack(cols, axis=-1).astype(np.float32)      # (n_patients, 4)
    block = np.repeat(per_patient[:, None, :], n_epochs, axis=1)
    return block, RTMS_FEATURE_NAMES


def eeg_block(
    dataset: LoadedDataset, per_patient_zscore: bool
) -> tuple[np.ndarray, tuple[str, ...]]:
    """``(n_patients, n_epochs, n_channels * n_bands)`` relative band powers."""
    from .tdbrain import montage_band_powers, montage_feature_names, zscore_epochs

    mc = dataset.signals_mc
    if mc is None:
        mc = dataset.signals[:, :, None, :]
        channels = dataset.channels or ["ch0"]
    else:
        channels = dataset.channels or [f"ch{i}" for i in range(mc.shape[2])]

    n_p, n_seq, n_ch, _ = mc.shape
    out = np.empty((n_p, n_seq, n_ch * len(BANDS)), dtype=np.float32)
    for p in range(n_p):
        out[p] = montage_band_powers(mc[p], dataset.fs)
    if per_patient_zscore and n_seq >= 2:
        out = zscore_epochs(out)
    return out, montage_feature_names(channels)


def ecg_block(dataset: LoadedDataset) -> tuple[np.ndarray, tuple[str, ...]]:
    """``(n_patients, n_epochs, 5)`` time-domain HRV features."""
    from ..preprocessing.features import hrv_features
    from ..preprocessing.pipeline import HRV_FEATURE_NAMES

    ecg = dataset.ecg
    if ecg is None:
        raise ModalityError(
            "modality 'ecg' requested but the dataset carries no RR tachogram"
        )
    n_p, n_seq, _ = ecg.shape
    out = np.empty((n_p, n_seq, len(HRV_FEATURE_NAMES)), dtype=np.float32)
    for p in range(n_p):
        for s in range(n_seq):
            feats = hrv_features(ecg[p, s])
            out[p, s] = [feats[name] for name in HRV_FEATURE_NAMES]
    return out, tuple(HRV_FEATURE_NAMES)


def build_features(
    dataset: LoadedDataset,
    modalities: tuple[str, ...] = ("eeg",),
    per_patient_zscore: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Assemble ``(x, y, groups, feature_names)`` for the requested modalities.

    Works identically on a real cohort from :func:`load_tdbrain` and a synthetic
    one from :func:`simulate_matched`.
    """
    unknown = [m for m in modalities if m not in MODALITY_ORDER]
    if unknown:
        raise ModalityError(
            f"unknown modalities {unknown}; expected any of {list(MODALITY_ORDER)}"
        )
    if not modalities:
        raise ModalityError("at least one modality is required")

    # Determine the sequence length from the signals, so the clinical block can be
    # broadcast to match even when 'eeg' is not requested.
    mc = dataset.signals_mc
    n_epochs = mc.shape[1] if mc is not None else dataset.signals.shape[1]

    blocks: list[np.ndarray] = []
    names: list[str] = []
    for modality in MODALITY_ORDER:            # canonical order, not caller order
        if modality not in modalities:
            continue
        if modality == "rtms":
            block, block_names = rtms_block(dataset, n_epochs)
        elif modality == "eeg":
            block, block_names = eeg_block(dataset, per_patient_zscore)
        else:
            block, block_names = ecg_block(dataset)
        blocks.append(block)
        names.extend(block_names)

    x = np.concatenate(blocks, axis=-1) if len(blocks) > 1 else blocks[0]

    y = dataset.labels.astype(np.int64)
    if "patient_id" in dataset.metadata:
        groups = dataset.metadata["patient_id"].to_numpy()
    else:
        groups = np.arange(x.shape[0])
    return x, y, groups, tuple(names)


def feature_dimension(dataset: LoadedDataset, modalities: tuple[str, ...]) -> int:
    """Width of the feature vector for ``modalities``, without building it."""
    n = 0
    if "rtms" in modalities:
        n += len(RTMS_FEATURE_NAMES)
    if "eeg" in modalities:
        mc = dataset.signals_mc
        n_ch = mc.shape[2] if mc is not None else 1
        n += n_ch * len(BANDS)
    if "ecg" in modalities:
        from ..preprocessing.pipeline import HRV_FEATURE_NAMES

        n += len(HRV_FEATURE_NAMES)
    return n
