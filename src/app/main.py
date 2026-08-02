from __future__ import annotations

import streamlit as st

from src.app.utils import (
    DataSource,
    db_is_empty,
    get_repository,
    has_trained_model,
    list_patient_ids,
    model_choice,
    source_config,
    source_selector,
)

st.set_page_config(
    page_title="NeuroRéponse — rTMS + LSTM",
    page_icon=":brain:",
    layout="wide",
)

st.title("NeuroRéponse — prédiction de la réponse au rTMS")
st.caption("PFE 2026 — Prédiction de la réponse au traitement rTMS à partir de signaux EEG.")

source = source_selector()
cfg_src = source_config(source)
repo = get_repository(source)
patient_ids = list_patient_ids(repo)

choix = model_choice(source)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Cohorte", cfg_src.label)
col2.metric("Patients en base", len(patient_ids))
col3.metric("Modèle entraîné", "oui" if has_trained_model(source) else "non")
col4.metric("Variables", choix.label)

st.divider()

if db_is_empty(repo) and source is DataSource.TDBRAIN:
    st.warning("La base TDBRAIN est vide.")
    st.write(
        "La cohorte réelle se charge en ligne de commande (lecture des fichiers BDF, "
        "quelques minutes) — elle n'est pas générable depuis l'interface :"
    )
    st.code(
        "python -m src.data.tdbrain_seeder "
        '--root "data/tdbrain/TDBRAIN_Dataset_V3_1_Encr/TDBRAIN_Dataset_V3_1" '
        f"--db {cfg_src.db}",
        language="powershell",
    )
elif db_is_empty(repo):
    st.warning("La base de données est vide.")
    quoi = (
        "la cohorte simulée appariée sur TDBRAIN (132 patients × 8 époques, "
        "26 canaux + ECG)"
        if source is DataSource.SIMULE
        else "le dataset simulé séquentiel (100 patients × 10 séances)"
    )
    st.write(f"Clique ci-dessous pour générer {quoi} et le charger dans SQLite.")
    if st.button("Initialiser cette cohorte"):
        from .utils import seed_demo_data

        with st.spinner("Génération + insertion en base…"):
            n = seed_demo_data(repo, source=source)
        st.success(f"Base initialisée avec {n} patients. Recharge la page.")
else:
    st.success(f"Base de données prête — {cfg_src.label}.")
    unit = f"{cfg_src.unit}s"
    st.write(
        "Utilise la barre latérale pour naviguer entre les pages :\n"
        "1. **Patients** — gestion des dossiers patients\n"
        f"2. **Sessions** — visualisation des {unit} et signaux EEG\n"
        "3. **Training** — entraînement du modèle LSTM (validation croisée patient-wise)\n"
        "4. **Predictions** — prédiction de la réponse + export PDF\n"
        "5. **Suivi** — évolution du patient sur toutes ses séances\n"
        "6. **Boucle clinique** — enregistrer, prédire, ajuster, recommencer\n"
        "7. **Comparaison** — les quatre modèles côte à côte"
    )
    if source is DataSource.SIMULE:
        st.info(
            "**Cohorte simulée appariée.** Générée pour reproduire la structure de "
            "TDBRAIN (âge, sexe, BDI-II, protocole, 26 canaux + ECG) sans effet "
            "neurophysiologique injecté : c'est le **contrôle négatif** de la "
            "comparaison 2×2, pas une cohorte au signal garanti."
        )
    if source is DataSource.TDBRAIN:
        st.info(
            "**Données réelles.** EEG, scores BDI-II et protocole rTMS proviennent de la "
            "cohorte TDBRAIN. Un seul enregistrement de repos par patient : les « séances » "
            "affichées sont des **époques** de cet enregistrement, pas une évolution du "
            "traitement."
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
