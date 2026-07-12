from __future__ import annotations

from pathlib import Path

import streamlit as st

from ..db import Repository

DB_PATH = Path("recherche.sqlite3")
MODEL_PATH = Path("data/models/lstm_v1.pt")
SIM_DIR = Path("data/simulated")


@st.cache_resource
def get_repository() -> Repository:
    return Repository(db_url=f"sqlite:///{DB_PATH}")


def list_patient_ids(repo: Repository) -> list[str]:
    from sqlalchemy import select

    from ..db.schema import PatientRow

    with repo._Session() as s:
        return list(s.execute(select(PatientRow.id).order_by(PatientRow.id)).scalars())


def db_is_empty(repo: Repository) -> bool:
    return len(list_patient_ids(repo)) == 0


def seed_demo_data(repo: Repository, limit: int | None = None) -> int:
    from ..data.loader import load
    from ..data.seeder import seed

    if not (SIM_DIR / "eeg_simulated.npz").exists():
        from ..data.simulator import save, simulate

        ds = simulate()
        save(ds, SIM_DIR)

    return seed(repo, load(SIM_DIR), limit=limit)


def has_trained_model() -> bool:
    return MODEL_PATH.exists()
