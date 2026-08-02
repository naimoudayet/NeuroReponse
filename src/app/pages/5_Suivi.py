"""Longitudinal follow-up: read every session at once, not just the last output.

Predictions answers "will this patient respond?". This page answers "how is this
patient going?" — clinical trajectory, model trajectory, and whether they agree —
computed from **all** the patient's sessions by :mod:`src.reporting.suivi`.

Under TDBRAIN the sessions are epochs of one baseline recording, so there is no
clinical trajectory to plot; the page says so and reports epoch-wise model
coherence instead of inventing progress.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from src.app.inference import build_model_input
from src.app.utils import (
    get_repository,
    has_trained_model,
    list_patient_ids,
    model_choice,
    model_path,
    source_config,
    source_selector,
)
from src.models.train import load_model
from src.models.train_tdbrain import load_contract
from src.reporting.suivi import SEUIL_REPONSE, analyser_suivi, trajectoire_scores

st.set_page_config(page_title="Suivi", page_icon=":chart_with_upwards_trend:", layout="wide")

source = source_selector()
cfg_src = source_config(source)
choix = model_choice(source)
repo = get_repository(source)
is_real = cfg_src.is_real
unite = cfg_src.unit

st.title("Suivi du patient" + (" (TDBRAIN)" if is_real else ""))
st.caption(
    f"Synthèse calculée sur **l'ensemble des {unite}s** du patient : trajectoire "
    f"clinique, trajectoire du modèle (TRI) et cohérence entre les deux."
)

patient_ids = list_patient_ids(repo)
if not patient_ids:
    st.info("Aucun patient en base.")
    st.stop()

pid = st.selectbox("Patient", patient_ids)
patient = repo.charger_patient(pid)
if patient is None or not patient.sessions:
    st.warning("Ce patient n'a aucune session.")
    st.stop()

# ----------------------------------------------------------------------------- #
# Model trajectory (optional): the TRI over every session.
# ----------------------------------------------------------------------------- #
tri: list[float] | None = None
model_warning: str | None = None

if has_trained_model(source) and (
    all(s.signaux for s in patient.sessions) or not choix.uses_signals
):
    try:
        model = load_model(model_path(source))
        contract = (
            load_contract(model_path(source)) if choix.requires_contract else None
        )
        if choix.requires_contract and contract is None:
            raise ValueError("contrat de features absent — ré-entraîne le modèle")
        x_input, _fs = build_model_input(patient, is_real, contract)
        with torch.no_grad():
            t = torch.as_tensor(x_input, dtype=torch.float32)
            tri = model.predict_tri(t).squeeze(0).cpu().numpy().tolist()
    except (KeyError, ValueError, RuntimeError) as exc:
        model_warning = str(exc)
else:
    model_warning = (
        "aucun modèle entraîné pour cette source"
        if not has_trained_model(source)
        else "certaines sessions n'ont pas de signal"
    )

syn = analyser_suivi(patient, tri_trajectory=tri, unite=unite)

# ----------------------------------------------------------------------------- #
# Headline metrics.
# ----------------------------------------------------------------------------- #
def fmt_pct(x: float | None) -> str:
    """Percent with just enough precision to stay truthful.

    A 29.0 -> 29.1 course is a slight *worsening*; at zero decimals that renders
    as the nonsensical "-0%". Values that round to zero but are not zero get a
    decimal place so the sign means something.
    """
    if x is None:
        return "—"
    return f"{x:.1%}" if 0 < abs(x) < 0.005 else f"{x:.0%}"


c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{unite.capitalize()}s", syn.n_sessions)
c2.metric("Score initial", f"{syn.score_initial:.1f}" if syn.score_initial is not None else "—")
c3.metric("Score final", f"{syn.score_final:.1f}" if syn.score_final is not None else "—")
c4.metric(
    "Réduction",
    fmt_pct(syn.reduction_pct),
    delta=(
        None if syn.repondeur_observe is None
        else ("répondeur" if syn.repondeur_observe else "non-répondeur")
    ),
    delta_color="normal" if syn.repondeur_observe else "inverse",
)

if not syn.trajectoire_clinique_disponible:
    st.warning(
        f"**Pas de trajectoire clinique.** Les {syn.n_sessions} {unite}s portent des "
        f"scores BDI-II identiques : TDBRAIN ne contient qu'un enregistrement de repos "
        f"avant traitement, découpé en {unite}s. Une évolution au fil du traitement "
        f"n'existe pas dans ces données et n'est donc pas affichée."
    )

# ----------------------------------------------------------------------------- #
# Clinical trajectory.
# ----------------------------------------------------------------------------- #
st.markdown("##### Trajectoire clinique (BDI-II)")
scores = [s for s in trajectoire_scores(patient) if s is not None]
if scores:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(scores) + 1)), y=scores, mode="lines+markers",
        name="Score post", line=dict(color="#3b82f6", width=3),
    ))
    if syn.score_initial is not None:
        fig.add_hline(
            y=syn.score_initial, line_dash="dot", line_color="#94a3b8",
            annotation_text=f"Base {syn.score_initial:.1f}", annotation_position="top left",
        )
        fig.add_hline(
            y=syn.score_initial * (1 - SEUIL_REPONSE), line_dash="dash", line_color="#22c55e",
            annotation_text=f"Seuil répondeur (−{SEUIL_REPONSE:.0%})",
            annotation_position="bottom left",
        )
    fig.update_layout(
        xaxis_title=f"{unite.capitalize()}", yaxis_title="BDI-II",
        height=340, margin=dict(l=40, r=20, t=20, b=40), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    if syn.tendance:
        st.caption(
            f"Pente estimée : {syn.pente_par_session:+.2f} point BDI-II par {unite} "
            f"→ **{syn.tendance}**."
        )
else:
    st.caption("Aucun score enregistré.")

# ----------------------------------------------------------------------------- #
# Model trajectory.
# ----------------------------------------------------------------------------- #
st.markdown(f"##### Trajectoire du modèle (TRI par {unite})")
if tri is None:
    st.info(f"Trajectoire du modèle indisponible : {model_warning}.")
else:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=list(range(1, len(tri) + 1)), y=tri, mode="lines+markers",
        name="TRI", line=dict(color="#a855f7", width=3),
    ))
    fig2.add_hline(y=SEUIL_REPONSE, line_dash="dash", line_color="#94a3b8",
                   annotation_text="Seuil 50%", annotation_position="bottom left")
    fig2.update_yaxes(range=[0, 1])
    fig2.update_layout(
        xaxis_title=f"{unite.capitalize()}", yaxis_title="P(réponse)",
        height=340, margin=dict(l=40, r=20, t=20, b=40), showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("TRI final", f"{syn.tri_final:.0%}")
    m2.metric("TRI moyen", f"{syn.tri_moyen:.0%}")
    m3.metric(f"Écart-type entre {unite}s", f"{syn.tri_ecart_type:.03f}")
    if is_real:
        st.caption(
            f"⚠️ Sur TDBRAIN l'axe est constitué d'**{unite}s d'un même "
            f"enregistrement** : cette courbe montre l'accumulation de preuve du "
            f"LSTM, pas une évolution clinique. L'écart-type mesure la cohérence "
            f"du modèle entre fenêtres, pas un progrès."
        )

# ----------------------------------------------------------------------------- #
# Synthesised feedback over all sessions.
# ----------------------------------------------------------------------------- #
st.divider()
st.markdown(f"##### Retour de synthèse — sur les {syn.n_sessions} {unite}s")
for msg in syn.messages:
    if msg.startswith("⚠️"):
        st.warning(msg)
    elif msg.startswith("❌"):
        st.error(msg)
    elif msg.startswith("✅"):
        st.success(msg)
    else:
        st.markdown(f"- {msg}")

with st.expander("Synthèse brute (analyser_suivi)"):
    st.json({k: v for k, v in syn.to_dict().items() if k != "messages"})

# ----------------------------------------------------------------------------- #
# Per-session table — the raw material behind the synthesis.
# ----------------------------------------------------------------------------- #
st.markdown(f"##### Détail par {unite}")
rows = []
for i, sess in enumerate(patient.sessions):
    rows.append({
        unite: i + 1,
        "id": sess.id_session,
        "date": sess.date.date(),
        "score_pre": sess.score_pre,
        "score_post": sess.score_post,
        "TRI": None if tri is None or i >= len(tri) else round(tri[i], 3),
        "nb_signaux": len(sess.signaux),
        "statut": sess.statut,
    })
st.dataframe(rows, hide_index=True, use_container_width=True)

if np.any([r["score_post"] is None for r in rows]):
    st.caption("Les scores manquants sont exclus du calcul de tendance.")
