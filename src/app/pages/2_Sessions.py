from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.app.utils import (
    format_rtms_parameters,
    get_repository,
    list_patient_ids,
    source_config,
    source_selector,
)
from src.domain import SignalType

st.set_page_config(page_title="Sessions", page_icon=":zap:", layout="wide")

source = source_selector()
cfg_src = source_config(source)
is_real = cfg_src.is_real
unit = cfg_src.unit.capitalize()

st.title("Époques du repos (TDBRAIN)" if is_real else "Sessions rTMS")
if is_real:
    st.caption(
        "TDBRAIN ne contient **qu'un enregistrement de repos avant traitement** par "
        "patient. Chaque « époque » ci-dessous est une fenêtre de ce même enregistrement : "
        "les paramètres rTMS et les scores BDI-II sont donc identiques d'une époque à "
        "l'autre — seul le signal EEG change."
    )

repo = get_repository(source)
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
sid = st.selectbox(unit, session_ids)
session = next(s for s in patient.sessions if s.id_session == sid)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Statut", session.statut)
c2.metric("Score pré-rTMS", f"{session.score_pre:.1f}" if session.score_pre is not None else "—")
c3.metric("Score post-rTMS", f"{session.score_post:.1f}" if session.score_post is not None else "—")
c4.metric("Nombre de signaux", len(session.signaux))

st.markdown("##### Paramètres rTMS")
params = session.parametres
st.write(format_rtms_parameters(params))
if is_real:
    st.caption(
        "TDBRAIN publie le **protocole** (fréquence + site) mais pas la dose de "
        "stimulation par patient : intensité, durée/nombre de trains et intervalle "
        "sont absents de la base source et affichés « non publié » plutôt que "
        "comblés par des valeurs plausibles."
    )

st.divider()
st.markdown("##### Signaux enregistrés")
if not session.signaux:
    st.caption("Aucun signal enregistré.")
else:
    signal_options = [f"{s.canal} ({s.type_signal.value})" for s in session.signaux]
    sig_idx = st.selectbox("Canal", range(len(signal_options)), format_func=lambda i: signal_options[i])
    signal = session.signaux[sig_idx]

    fig = go.Figure()
    if signal.type_signal is SignalType.ECG:
        # An RR tachogram is event-sampled: x is the beat index, not seconds, and
        # dividing by sampling_rate_hz (0.0 by construction) would blow up.
        rr_ms = np.asarray(signal.valeurs, dtype=float) * 1000.0
        fig.add_trace(go.Scatter(
            x=np.arange(len(rr_ms)), y=rr_ms, mode="lines+markers",
            name="RR", line=dict(color="#ef4444"),
        ))
        fig.update_layout(
            title=f"Tachogramme RR — dérivation {signal.canal} ({len(rr_ms)} battements)",
            xaxis_title="Battement (index)",
            yaxis_title="Intervalle RR (ms)",
        )
    else:
        t = np.arange(len(signal.valeurs)) / signal.sampling_rate_hz
        fig.add_trace(go.Scatter(
            x=t, y=signal.valeurs, mode="lines",
            name=signal.type_signal.value, line=dict(color="#3b82f6"),
        ))
        fig.update_layout(
            title=(f"Signal {signal.type_signal.value} — canal {signal.canal} "
                   f"({signal.sampling_rate_hz} Hz)"),
            xaxis_title="Temps (s)",
            yaxis_title="Amplitude",
        )
    fig.update_layout(height=350, margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)
    if signal.type_signal is SignalType.ECG:
        st.caption(
            "HRV mesurée sur l'enregistrement complet (~2 min) puis répétée sur "
            "chaque époque : c'est un trait **par patient**, pas une évolution "
            "temporelle. Une époque de 8 s ne contient que ~9 battements."
        )

    with st.expander("Features extraits"):
        feats = signal.extraire_features()
        st.json(feats)

st.divider()
with st.expander("Rapport de séance (generer_rapport)"):
    st.json(session.generer_rapport())
