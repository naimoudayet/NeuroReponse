from __future__ import annotations

import streamlit as st

from src.app.utils import (
    db_is_empty,
    get_repository,
    has_trained_model,
    list_patient_ids,
)

st.set_page_config(
    page_title="rTMS + LSTM — Recherche App",
    page_icon=":brain:",
    layout="wide",
)

st.title("rTMS + LSTM — application de recherche")
st.caption("PFE 2026 — Prédiction de la réponse au traitement rTMS à partir de signaux EEG simulés.")

repo = get_repository()
patient_ids = list_patient_ids(repo)

col1, col2, col3 = st.columns(3)
col1.metric("Patients en base", len(patient_ids))
col2.metric("Modèle entraîné", "oui" if has_trained_model() else "non")
col3.metric("Backend ML", "PyTorch (CPU)")

st.divider()

if db_is_empty(repo):
    st.warning("La base de données est vide.")
    st.write(
        "Clique sur le bouton ci-dessous pour générer le dataset simulé (100 patients × "
        "10 séances) et le charger dans la base SQLite."
    )
    if st.button("Initialiser avec les données simulées"):
        from .utils import seed_demo_data

        with st.spinner("Génération + insertion en base…"):
            n = seed_demo_data(repo)
        st.success(f"Base initialisée avec {n} patients. Recharge la page.")
else:
    st.success("Base de données prête.")
    st.write(
        "Utilise la barre latérale pour naviguer entre les pages :\n"
        "1. **Patients** — gestion des dossiers patients\n"
        "2. **Sessions** — visualisation des séances rTMS et signaux EEG\n"
        "3. **Training** — entraînement du modèle LSTM (validation croisée patient-wise)\n"
        "4. **Predictions** — prédiction de la réponse + export PDF"
    )

st.divider()
with st.expander("Architecture (mapping UML → code)"):
    st.markdown(
        """
        | UML class                | Module Python                                  |
        |--------------------------|-----------------------------------------------|
        | Patient                  | `src/domain/patient.py`                        |
        | SessionRTMS              | `src/domain/session_rtms.py`                   |
        | SignalNeurophysiologique | `src/domain/signal_neuro.py`                   |
        | Preprocessing            | `src/domain/preprocessing.py` + `src/preprocessing/pipeline.py` |
        | ModeleLSTM               | `src/domain/lstm_model.py` + `src/models/`     |
        | Prediction               | `src/domain/prediction.py`                     |
        | ClinicianInterface       | `src/domain/clinician.py` + cette app Streamlit |
        | BaseDeDonnées            | `src/db/repository.py` (SQLAlchemy + SQLite)   |
        """
    )
