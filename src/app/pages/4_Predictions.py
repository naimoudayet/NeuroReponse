from __future__ import annotations

from datetime import datetime
from io import BytesIO

import numpy as np
import streamlit as st
import torch

from src.app.utils import (
    MODEL_PATH,
    get_repository,
    has_trained_model,
    list_patient_ids,
)
from src.domain import ClinicianInterface, Prediction
from src.domain.clinician import Role
from src.models.train import load_model
from src.preprocessing.pipeline import PipelineConfig, preprocess

st.set_page_config(page_title="Predictions", page_icon=":crystal_ball:", layout="wide")
st.title("Prédiction de la réponse au traitement")

if not has_trained_model():
    st.error("Aucun modèle entraîné. Va sur la page **Training** d'abord.")
    st.stop()

repo = get_repository()
patient_ids = list_patient_ids(repo)
if not patient_ids:
    st.info("Aucun patient en base.")
    st.stop()

pid = st.selectbox("Patient", patient_ids)
patient = repo.charger_patient(pid)

if patient is None or not patient.sessions:
    st.warning("Ce patient n'a pas de séances → impossible de prédire.")
    st.stop()


@st.cache_resource
def _load_model(path_str: str):
    return load_model(MODEL_PATH)


model = _load_model(str(MODEL_PATH))
fs = patient.sessions[0].signaux[0].sampling_rate_hz if patient.sessions[0].signaux else 256.0

if not all(s.signaux for s in patient.sessions):
    st.warning("Certaines séances n'ont pas de signal EEG → impossible de prédire.")
    st.stop()

window = patient.sessions[0].signaux[0].valeurs.shape[0]
signals = np.stack(
    [np.stack([sig.valeurs[:window] for sig in sess.signaux[:1]]).squeeze(0)
     for sess in patient.sessions]
)
signals_3d = signals[np.newaxis, :, :]
pre = preprocess(signals_3d.astype(np.float32), PipelineConfig(fs=fs, mode="features"))

x_tensor = torch.as_tensor(pre.x, dtype=torch.float32)
with torch.no_grad():
    proba = float(model.predict_proba(x_tensor).item())
    tri_traj = model.predict_tri(x_tensor).squeeze(0).cpu().numpy().tolist()

valeur = int(proba >= 0.5)
prediction = Prediction(
    patient_id=patient.id,
    valeur=valeur,
    probabilite=proba,
    date=datetime.now(),
    model_version="lstm_v1",
    tri_trajectory=tri_traj,
)

c1, c2, c3 = st.columns(3)
c1.metric("Probabilité de réponse", f"{proba:.1%}")
c2.metric("Classe prédite", "Répondeur" if valeur == 1 else "Non-répondeur")
c3.metric("Sessions utilisées", len(patient.sessions))

st.write(f"**Interprétation** — {prediction.afficher()}")

st.markdown("##### Indice thérapeutique (TRI) — trajectoire par séance")
st.caption(
    "TRIₜ = σ(W·hₜ + b) : estimation de la probabilité de réponse au fil des séances, "
    "à mesure que le LSTM accumule les preuves (la dernière valeur = probabilité finale)."
)
st.line_chart(
    {"TRI": tri_traj},
    x=None,
    height=240,
)

if patient.historique_clinique:
    last_score = patient.historique_clinique[-1].score_depression
    if last_score is not None:
        ecart = prediction.analyser_ecart(last_score / 30.0)
        st.markdown("##### Comparaison au score clinique observé")
        st.json(ecart)

st.divider()
left, right = st.columns(2)

with left:
    if st.button("Sauvegarder cette prédiction en base"):
        pred_id = repo.sauvegarder_prediction(prediction)
        st.success(f"Prédiction sauvegardée (id={pred_id}).")

    history = repo.lister_predictions(patient.id)
    if history:
        st.markdown("##### Historique des prédictions")
        st.dataframe(
            [{"date": p.date, "classe": p.valeur, "probabilité": p.probabilite, "modèle": p.model_version}
             for p in history],
            hide_index=True, use_container_width=True,
        )

with right:
    ci = ClinicianInterface(nom_utilisateur="dr_demo", role=Role.CLINICIEN)
    pdf_buf = BytesIO()
    ci.exporter_pdf([prediction], pdf_buf)
    pdf_buf.seek(0)
    st.download_button(
        "Télécharger le rapport PDF",
        data=pdf_buf,
        file_name=f"prediction_{patient.id}.pdf",
        mime="application/pdf",
    )
