"""Modality ablation for the multimodal (EEG + ERP + ECG) response model.

Reviewers of the NPDT design (RES0_AR1) will ask the obvious question: does each
extra modality actually earn its place? This runs the *same* patient-wise
GroupKFold CV for each modality subset and returns a comparison table, so the
"multimodal > unimodal" claim is demonstrated rather than asserted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..preprocessing.pipeline import MultimodalConfig, preprocess_multimodal
from .lstm import LSTMConfig
from .train import TrainConfig, cross_validate

DEFAULT_MODALITY_SETS: tuple[tuple[str, ...], ...] = (
    ("eeg",),
    ("eeg", "erp"),
    ("eeg", "ecg"),
    ("eeg", "erp", "ecg"),
)


@dataclass
class AblationRow:
    modalities: str
    n_features: int
    accuracy_mean: float
    accuracy_std: float
    auc_mean: float
    auc_std: float
    f1_mean: float
    f1_std: float


def run_ablation(
    eeg: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    erp: np.ndarray | None = None,
    ecg: np.ndarray | None = None,
    fs: float = 256.0,
    modality_sets: tuple[tuple[str, ...], ...] = DEFAULT_MODALITY_SETS,
    dropout: float = 0.3,
    train_cfg: TrainConfig | None = None,
    n_splits: int = 5,
) -> list[AblationRow]:
    """Train + evaluate one model per modality subset; return their summary metrics."""
    train_cfg = train_cfg or TrainConfig()
    y = labels.astype(np.float32)
    rows: list[AblationRow] = []

    for modalities in modality_sets:
        pre = preprocess_multimodal(eeg, erp, ecg, MultimodalConfig(fs=fs, modalities=modalities))
        cv = cross_validate(
            pre.x, y, groups,
            lstm_cfg=LSTMConfig(input_size=pre.x.shape[-1], dropout=dropout),
            train_cfg=train_cfg,
            n_splits=n_splits,
        )
        s = cv.summary()
        rows.append(
            AblationRow(
                modalities="+".join(modalities),
                n_features=int(pre.x.shape[-1]),
                accuracy_mean=s["accuracy_mean"], accuracy_std=s["accuracy_std"],
                auc_mean=s["auc_mean"], auc_std=s["auc_std"],
                f1_mean=s["f1_mean"], f1_std=s["f1_std"],
            )
        )
    return rows


def ablation_dataframe(rows: list[AblationRow]):
    """Convenience: turn ablation rows into a pandas DataFrame for notebooks/app."""
    import pandas as pd

    return pd.DataFrame([r.__dict__ for r in rows])
