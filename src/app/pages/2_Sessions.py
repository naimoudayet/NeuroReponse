from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.app.utils import get_repository, list_patient_ids

st.set_page_config(page_title="Sessions", page_icon=":zap:", layout="wide")
st.title("Sessions rTMS")

repo = get_repository()
patient_ids = list_patient_ids(repo)

if not patient_ids:
    st.info("Aucun patient en base. Va sur l'accueil pour initialiser.")
    st.stop()

pid = st.selectbox("Patient", patient_ids)
patient = repo.charger_patient(pid)

if patient is None or not patient.sessions:
    st.warning("Ce patient n'a aucune séance enregistrée.")
    st.stop()

session_ids = [s.id_session for s in patient.sessions]
sid = st.selectbox("Séance", session_ids)
session = next(s for s in patient.sessions if s.id_session == sid)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Statut", session.statut)
c2.metric("Score pré-rTMS", f"{session.score_pre:.1f}" if session.score_pre is not None else "—")
c3.metric("Score post-rTMS", f"{session.score_post:.1f}" if session.score_post is not None else "—")
c4.metric("Nombre de signaux", len(session.signaux))

st.markdown("##### Paramètres rTMS")
params = session.parametres
st.write({
    "Fréquence (Hz)": params.frequence_hz,
    "Intensité (%)": params.intensite_pct,
    "Durée train (s)": params.duree_train_s,
    "Nb trains": params.nb_trains,
    "Localisation": params.localisation,
    "Protocole": params.protocole,
    "Total pulses": params.total_pulses(),
})

st.divider()
st.markdown("##### Signaux EEG")
if not session.signaux:
    st.caption("Aucun signal enregistré.")
else:
    signal_options = [f"{s.canal} ({s.type_signal.value})" for s in session.signaux]
    sig_idx = st.selectbox("Canal", range(len(signal_options)), format_func=lambda i: signal_options[i])
    signal = session.signaux[sig_idx]

    t = np.arange(len(signal.valeurs)) / signal.sampling_rate_hz
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=signal.valeurs, mode="lines", name="EEG", line=dict(color="#3b82f6")))
    fig.update_layout(
        title=f"Signal {signal.type_signal.value} — canal {signal.canal} ({signal.sampling_rate_hz} Hz)",
        xaxis_title="Temps (s)",
        yaxis_title="Amplitude",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Features extraits"):
        feats = signal.extraire_features()
        st.json(feats)

st.divider()
with st.expander("Rapport de séance (generer_rapport)"):
    st.json(session.generer_rapport())
