from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from src.app.utils import (
    DataSource,
    get_repository,
    list_patient_ids,
    seed_demo_data,
    source_config,
    source_selector,
)
from src.domain import Patient
from src.domain.patient import DossierClinique

st.set_page_config(page_title="Patients", page_icon=":bust_in_silhouette:", layout="wide")
st.title("Patients")

source = source_selector()
repo = get_repository(source)
patient_ids = list_patient_ids(repo)

with st.sidebar:
    st.subheader("Actions")
    if source is DataSource.TDBRAIN:
        st.caption(
            "Cohorte réelle — chargement via "
            "`python -m src.data.tdbrain_seeder`."
        )
    elif st.button("Recharger les données simulées"):
        with st.spinner("Insertion…"):
            n = seed_demo_data(repo, source=source)
        st.success(f"{n} patients chargés.")
        st.rerun()

if not patient_ids:
    st.info("Aucun patient en base. Va sur l'accueil pour initialiser, ou utilise la barre latérale.")
    st.stop()

st.subheader(f"Liste ({len(patient_ids)} patients)")

rows = []
for pid in patient_ids:
    p = repo.charger_patient(pid)
    if p is None:
        continue
    last_score = (
        p.historique_clinique[-1].score_depression
        if p.historique_clinique else None
    )
    rows.append({
        "id": p.id,
        "nom": p.nom,
        "âge": round(float(p.age), 1),
        "sexe": "—" if p.sexe is None else p.sexe,
        "diagnostic": p.diagnostic,
        "nb_sessions": len(p.sessions),
        "dernier_score": last_score,
    })

df = pd.DataFrame(rows)
st.dataframe(df, hide_index=True, use_container_width=True)

st.divider()
st.subheader("Détail d'un patient")
selected = st.selectbox("Patient", patient_ids, index=0)
patient = repo.charger_patient(selected)

if patient:
    c1, c2 = st.columns(2)
    c1.markdown(f"**Nom** : {patient.nom}")
    c1.markdown(f"**Âge** : {float(patient.age):.1f}")
    c1.markdown(
        "**Sexe** : "
        + ("— (non renseigné)" if patient.sexe is None else str(patient.sexe))
    )
    c1.markdown(f"**Diagnostic** : {patient.diagnostic}")
    c2.markdown(f"**Nombre de séances** : {len(patient.sessions)}")
    c2.markdown(f"**Entrées au dossier** : {len(patient.historique_clinique)}")

    st.markdown("##### Historique clinique")
    if patient.historique_clinique:
        hist_df = pd.DataFrame([
            {"date": d.date.date(), "note": d.note, "score": d.score_depression}
            for d in patient.historique_clinique
        ])
        st.dataframe(hist_df, hide_index=True, use_container_width=True)
    else:
        st.caption("Aucune entrée.")

    with st.expander("Ajouter une entrée au dossier"):
        with st.form(f"add_dossier_{patient.id}", clear_on_submit=True):
            note = st.text_area("Note clinique")
            score = st.number_input("Score de dépression (HDRS-like)", min_value=0.0, max_value=60.0, step=0.5)
            submitted = st.form_submit_button("Ajouter")
            if submitted and note:
                patient.ajouter_dossier(DossierClinique(date=datetime.now(), note=note, score_depression=score))
                repo.sauvegarder_patient(patient)
                st.success("Entrée ajoutée.")
                st.rerun()

st.divider()
_cfg_src = source_config(source)
with st.expander("Créer un nouveau patient"):
    if source is DataSource.TDBRAIN:
        st.caption(
            "⚠️ Un patient créé ici n'aura aucun signal EEG : il apparaîtra dans la liste "
            "mais ne pourra pas être utilisé pour la prédiction."
        )
    with st.form("new_patient", clear_on_submit=True):
        new_id = st.text_input("ID patient (ex: P999)")
        new_nom = st.text_input("Nom")
        new_age = st.number_input("Âge", min_value=0.0, max_value=120.0, value=40.0, step=0.5)
        new_sexe = st.selectbox(
            "Sexe", [0, 1],
            format_func=lambda v: f"{v} — {'femme' if v == 0 else 'homme'}",
            help=(
                "Codé 0/1 comme dans la table source. Le bloc clinique des quatre "
                "modèles le lit : sans lui, aucune prédiction n'est possible."
            ),
        )
        new_diag = st.text_input("Diagnostic", value="MDD")
        submitted = st.form_submit_button("Créer")
        if submitted and new_id and new_nom:
            p = Patient(id=new_id, nom=new_nom, age=float(new_age),
                        sexe=int(new_sexe), diagnostic=new_diag)
            repo.sauvegarder_patient(p)
            st.success(f"Patient {new_id} créé.")
            st.rerun()
