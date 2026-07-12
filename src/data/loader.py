from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class LoadedDataset:
    signals: np.ndarray   # (n_patients, n_sessions, window)  — EEG
    labels: np.ndarray    # (n_patients,)
    fs: float
    window: int
    metadata: pd.DataFrame
    erp: np.ndarray | None = None   # (n_patients, n_sessions, window)  — evoked potential
    ecg: np.ndarray | None = None   # (n_patients, n_sessions, n_rr)    — RR intervals (s)


def load(data_dir: Path = Path("data/simulated")) -> LoadedDataset:
    npz_path = data_dir / "eeg_simulated.npz"
    csv_path = data_dir / "metadata.csv"

    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Generate it first: python -m src.data.simulator"
        )

    with np.load(npz_path) as npz:
        signals = npz["signals"]
        labels = npz["labels"]
        fs = float(npz["fs"])
        window = int(npz["window"])
        erp = npz["erp"] if "erp" in npz.files else None
        ecg = npz["ecg"] if "ecg" in npz.files else None

    metadata = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    return LoadedDataset(
        signals=signals, labels=labels, fs=fs, window=window,
        metadata=metadata, erp=erp, ecg=ecg,
    )
