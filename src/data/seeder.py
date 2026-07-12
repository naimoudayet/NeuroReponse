from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ..db import Repository
from ..domain import (
    Patient,
    RTMSParameters,
    SessionRTMS,
    SignalNeurophysiologique,
    SignalType,
)
from ..domain.patient import DossierClinique
from .loader import LoadedDataset, load


def _patient_from_simulated(
    p_idx: int,
    label: int,
    sessions_signals: np.ndarray,
    metadata_rows,
    fs: float,
) -> Patient:
    patient_id = f"P{p_idx:03d}"
    patient = Patient(
        id=patient_id,
        nom=f"Patient {p_idx:03d}",
        age=int(40 + (p_idx % 25)),
        diagnostic="MDD (simulé)",
    )

    base_date = datetime(2026, 1, 1)
    for s_idx, signal_window in enumerate(sessions_signals):
        meta = metadata_rows.iloc[s_idx]
        params = RTMSParameters(
            frequence_hz=float(meta["frequence_hz"]),
            intensite_pct=float(meta["intensite_pct"]),
            duree_train_s=4.0,
            nb_trains=75,
            intervalle_train_s=26.0,
            localisation=str(meta["localisation"]),
            protocole="standard_depression",
        )
        session = SessionRTMS(
            id_session=f"{patient_id}-S{s_idx:02d}",
            patient_id=patient_id,
            parametres=params,
            date=base_date + timedelta(days=s_idx * 2),
        )
        session.demarrer()
        session.enregistrer_donnees(
            SignalNeurophysiologique(
                type_signal=SignalType.EEG,
                valeurs=signal_window.astype(np.float32),
                timestamp=session.date,
                canal="Fz",
                sampling_rate_hz=fs,
            )
        )
        session.cloturer(score_post=float(meta["score_clinique"]))
        session.score_pre = float(metadata_rows.iloc[0]["score_clinique"])
        patient.ajouter_session(session)

    patient.ajouter_dossier(
        DossierClinique(
            date=base_date,
            note="Dossier généré automatiquement (données simulées).",
            score_depression=float(metadata_rows.iloc[0]["score_clinique"]),
        )
    )
    patient.ajouter_dossier(
        DossierClinique(
            date=base_date + timedelta(days=20),
            note=("Répondeur" if label == 1 else "Non-répondeur") + " (label simulé)",
            score_depression=float(metadata_rows.iloc[-1]["score_clinique"]),
        )
    )
    return patient


def seed(repo: Repository, dataset: LoadedDataset | None = None, limit: int | None = None) -> int:
    dataset = dataset or load()
    n = dataset.signals.shape[0] if limit is None else min(limit, dataset.signals.shape[0])
    count = 0
    for p_idx in range(n):
        rows = dataset.metadata[dataset.metadata["patient_id"] == f"P{p_idx:03d}"]
        if rows.empty:
            continue
        patient = _patient_from_simulated(
            p_idx=p_idx,
            label=int(dataset.labels[p_idx]),
            sessions_signals=dataset.signals[p_idx],
            metadata_rows=rows.reset_index(drop=True),
            fs=dataset.fs,
        )
        repo.sauvegarder_patient(patient)
        count += 1
    return count


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Seed SQLite DB from simulated dataset.")
    ap.add_argument("--db", type=Path, default=Path("recherche.sqlite3"))
    ap.add_argument("--data", type=Path, default=Path("data/simulated"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dataset = load(args.data)
    repo = Repository(db_url=f"sqlite:///{args.db}")
    n = seed(repo, dataset=dataset, limit=args.limit)
    print(f"Seeded {n} patients into {args.db}")


if __name__ == "__main__":
    _main()
