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

``sync``
    30 network features — PLV, coherence and the Kuramoto order parameter, five
    bands x six summaries. This is the block the reference study has no
    equivalent of: band power is blind to phase, so nothing in a 130-column
    power vector can express *how* two channels relate.

``cplx``
    6 complexity features — spectral entropy, the 1/f aperiodic exponent (the
    tractable stand-in for the Wilson-Cowan E/I state), frontal alpha asymmetry
    and the individual alpha frequency.

``h369``
    4 harmonic ratios testing the falsifiable Tesla 3-6-9 hypothesis.

The last three come from :mod:`src.preprocessing.connectivity`; see that module
for why each one is aggregated down to a handful of columns rather than emitted
per channel pair.

**Two blocks are patient-level, and that governs normalisation.** The clinical and
HRV blocks are constant along the epoch axis by construction, so per-patient
z-scoring across epochs would divide by a zero standard deviation and collapse
them to constant zero — deleting the modality while every shape assertion still
passes. Only the EEG block is z-scored. ``tests/test_modalities.py`` pins this.

The three network blocks *do* vary across epochs, so per-patient z-scoring would
not blank them — but it is still not applied, and the reason is measured rather
than stylistic. ``src.reporting.effect_sweep`` shows a planted between-patient
effect climbing 0.582 -> 0.730 in AUC on raw band power and staying dead flat
(0.468 -> 0.463) on the z-scored version of the same features, because centring
each patient on their own epochs subtracts precisely the quantity that carries a
between-patient label. Responder status is such a label. Fold-safe *cohort*
standardisation is available instead, in ``cross_validate(standardise=True)``.

**Block order is canonical, not argument order.** Blocks are always concatenated
as ``rtms, eeg, ecg, sync, cplx, h369`` regardless of how ``modalities`` is
spelled, so a checkpoint can never be fed a permuted feature vector — the same
class of bug that selecting EEG channels by name (rather than row order) prevents
elsewhere. The three new blocks are **appended** rather than slotted next to
``eeg`` where they belong thematically: every checkpoint trained before they
existed must keep receiving a byte-identical vector, and only appending
guarantees that.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..preprocessing.features import BANDS
from .loader import LoadedDataset

MODALITY_ORDER: tuple[str, ...] = ("rtms", "eeg", "ecg", "sync", "cplx", "h369")

# The blocks that need the full montage waveform rather than a summary already
# stored in the dataset. Used to fail early with a readable message instead of
# indexing `signals_mc` when it is None.
_MONTAGE_BLOCKS: frozenset[str] = frozenset({"sync", "cplx", "h369"})

# What the model is asked to predict.
#   "responder"     — binary, >=50% BDI-II reduction. This project's original task.
#   "delta_bdi"     — BDI-II points recovered. What Arteaga et al. (PMC12981298)
#                     regress, and therefore the only target comparable to their r.
#   "pct_reduction" — the same change relative to baseline severity.
#
# The two continuous targets are NOT interchangeable. `delta_bdi` is
# mathematically coupled to baseline severity (you cannot drop 40 points from a
# BDI of 20): measured on TDBRAIN protocol 1, `bdi_pre` alone correlates with it
# at r = 0.500, the same magnitude as the article's headline r = 0.401 from EEG
# (bootstrap CI [0.258, 0.700] on 44 patients, so indistinguishable from it).
# `pct_reduction` divides that coupling out (r = 0.156). Report both, and always against the
# clinical-only baseline; see `src.models.metrics.baseline_report`.
TARGETS: tuple[str, ...] = ("responder", "delta_bdi", "pct_reduction")

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


def _montage(dataset: LoadedDataset, modality: str) -> tuple[np.ndarray, list[str]]:
    """``(signals_mc, channel names)``, or a readable error.

    The network blocks are the first ones in this project that genuinely cannot
    be built from a single representative channel: PLV, coherence and the
    Kuramoto order parameter are all defined *between* channels. Falling back to
    ``signals[:, :, None, :]`` the way :func:`eeg_block` does would silently
    return a matrix of ones (every pair is the channel with itself).
    """
    mc = dataset.signals_mc
    if mc is None or mc.shape[2] < 2:
        raise ModalityError(
            f"modality {modality!r} needs a multi-channel montage; this dataset "
            f"carries {'none' if mc is None else mc.shape[2]} channel(s). "
            "Connectivity between one channel and itself is not defined."
        )
    channels = dataset.channels or [f"ch{i}" for i in range(mc.shape[2])]
    return mc, list(channels)


def sync_block(dataset: LoadedDataset) -> tuple[np.ndarray, tuple[str, ...]]:
    """``(n_patients, n_epochs, 30)`` PLV / coherence / Kuramoto features."""
    from ..preprocessing.connectivity import montage_sync_features, sync_metric_names

    mc, channels = _montage(dataset, "sync")
    out = np.stack([
        montage_sync_features(mc[p], dataset.fs, channels) for p in range(mc.shape[0])
    ]).astype(np.float32)
    return out, sync_metric_names()


def complexity_block(dataset: LoadedDataset) -> tuple[np.ndarray, tuple[str, ...]]:
    """``(n_patients, n_epochs, 6)`` entropy / 1-f slope / alpha asymmetry."""
    from ..preprocessing.connectivity import (
        COMPLEXITY_METRICS,
        montage_complexity_features,
    )

    mc, channels = _montage(dataset, "cplx")
    out = np.stack([
        montage_complexity_features(mc[p], dataset.fs, channels)
        for p in range(mc.shape[0])
    ]).astype(np.float32)
    return out, COMPLEXITY_METRICS


def h369_block(dataset: LoadedDataset) -> tuple[np.ndarray, tuple[str, ...]]:
    """``(n_patients, n_epochs, 4)`` Tesla 3-6-9 harmonic ratios."""
    from ..preprocessing.connectivity import H369_METRICS, montage_h369_features

    mc, _ = _montage(dataset, "h369")
    out = np.stack([
        montage_h369_features(mc[p], dataset.fs) for p in range(mc.shape[0])
    ]).astype(np.float32)
    return out, H369_METRICS


def target_values(dataset: LoadedDataset, target: str = "responder") -> np.ndarray:
    """The outcome vector for ``target``, one value per patient.

    ``delta_bdi`` is derived from ``bdi_pre``/``bdi_post`` when the column is
    absent — the database round-trip keeps those two but not their difference.
    """
    if target not in TARGETS:
        raise ModalityError(f"unknown target {target!r}; expected any of {list(TARGETS)}")
    if target == "responder":
        return dataset.labels.astype(np.int64)

    meta = dataset.metadata
    if target == "delta_bdi" and "delta_bdi" not in meta:
        if not {"bdi_pre", "bdi_post"} <= set(meta.columns):
            raise ModalityError(
                "target 'delta_bdi' requires bdi_pre/bdi_post in the metadata"
            )
        return (meta["bdi_pre"].to_numpy(float) - meta["bdi_post"].to_numpy(float))
    if target not in meta:
        raise ModalityError(f"target {target!r} absent from the dataset metadata")
    return meta[target].to_numpy(dtype=np.float64)


def protocol_mask(dataset: LoadedDataset, protocol: int | None) -> np.ndarray:
    """Boolean row mask selecting one rTMS protocol, or everything for ``None``.

    The two protocols are different treatments — 10 Hz excitatory over the left
    DLPFC versus 1 Hz inhibitory over the right — and the reference study models
    them with two separate models rather than one pooled one. Pooling them is
    still offered (``None``) so the cost of pooling can be measured, not assumed.
    """
    n = dataset.signals.shape[0]
    if protocol is None:
        return np.ones(n, dtype=bool)
    if "protocol" not in dataset.metadata:
        raise ModalityError(
            "protocol filtering requested but the dataset metadata has no "
            "'protocol' column"
        )
    mask = dataset.metadata["protocol"].to_numpy() == protocol
    if not mask.any():
        raise ModalityError(f"no patient with rTMS protocol {protocol}")
    return np.asarray(mask, dtype=bool)


def build_features(
    dataset: LoadedDataset,
    modalities: tuple[str, ...] = ("eeg",),
    per_patient_zscore: bool = True,
    target: str = "responder",
    protocol: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Assemble ``(x, y, groups, feature_names)`` for the requested modalities.

    Works identically on a real cohort from :func:`load_tdbrain` and a synthetic
    one from :func:`simulate_matched`.

    ``protocol`` restricts to one rTMS arm; ``target`` picks the outcome. Both
    filters are applied **after** the feature blocks are built, so per-patient
    z-scoring is unaffected (it never crossed patients) while the cohort-level
    statistics a protocol subset changes are computed on the subset.
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
        elif modality == "ecg":
            block, block_names = ecg_block(dataset)
        elif modality == "sync":
            block, block_names = sync_block(dataset)
        elif modality == "cplx":
            block, block_names = complexity_block(dataset)
        else:
            block, block_names = h369_block(dataset)
        blocks.append(block)
        names.extend(block_names)

    x = np.concatenate(blocks, axis=-1) if len(blocks) > 1 else blocks[0]

    y = target_values(dataset, target)
    if "patient_id" in dataset.metadata:
        groups = dataset.metadata["patient_id"].to_numpy()
    else:
        groups = np.arange(x.shape[0])

    mask = protocol_mask(dataset, protocol)
    if not mask.all():
        x, y, groups = x[mask], y[mask], groups[mask]
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
    if _MONTAGE_BLOCKS & set(modalities):
        from ..preprocessing.connectivity import (
            COMPLEXITY_METRICS,
            H369_METRICS,
            sync_metric_names,
        )

        if "sync" in modalities:
            n += len(sync_metric_names())
        if "cplx" in modalities:
            n += len(COMPLEXITY_METRICS)
        if "h369" in modalities:
            n += len(H369_METRICS)
    return n
