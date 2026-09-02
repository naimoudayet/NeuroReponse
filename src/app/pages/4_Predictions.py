from __future__ import annotations

from datetime import datetime
from io import BytesIO

import streamlit as st
import torch

from src.app.inference import build_model_input, eeg_sampling_rate
from src.app.utils import (
    PROTOCOLES,
    current_protocol,
    get_repository,
    has_trained_model,
    list_patient_ids,
    model_choice,
    model_path,
    source_config,
    source_selector,
)
from src.domain import ClinicianInterface, Prediction
from src.domain.clinician import Role
from src.models.train import load_model
from src.models.train_tdbrain import load_contract

st.set_page_config(page_title="Predictions", page_icon=":crystal_ball:", layout="wide")
st.title("Prédiction de la réponse au traitement")

source = source_selector()
cfg = source_config(source)
choix = model_choice(source)
is_real = cfg.is_real
unit = cfg.unit

protocole = current_protocol(source)
st.caption(
    f"**{cfg.label}** · modèle **{choix.label}** · "
    f"cible **{'réduction BDI-II (ΔBDI)' if choix.is_regression else 'réponse binaire'}**"
    + (f" · {PROTOCOLES[protocole]}" if protocole is not None else "")
)

if not has_trained_model(source):
    st.error(
        f"Aucun modèle entraîné pour **{cfg.label} · {choix.label}** "
        f"(`{model_path(source)}` absent). Va sur la page **Training** d'abord."
    )
    st.stop()

repo = get_repository(source)
# Filtered on the model's own arm: a checkpoint fitted on protocol 1 must never be
# offered a protocol-2 patient, and the feature vector is the right shape either
# way, so nothing downstream would catch it.
patient_ids = list_patient_ids(repo, protocole)
if not patient_ids:
    st.info(f"Aucun patient en base pour **{cfg.label}**.")
    st.stop()
if protocole is not None:
    st.caption(f"{len(patient_ids)} patients dans ce bras de traitement.")

pid = st.selectbox("Patient", patient_ids)
patient = repo.charger_patient(pid)

if patient is None or not patient.sessions:
    st.warning(f"Ce patient n'a pas de {unit}s → impossible de prédire.")
    st.stop()
if choix.uses_signals and not all(s.signaux for s in patient.sessions):
    st.warning(f"Certaines {unit}s n'ont pas de signal EEG → impossible de prédire.")
    st.stop()


@st.cache_resource
def _load_model(path_str: str):
    return load_model(model_path(source))


model = _load_model(str(model_path(source)))
contract = load_contract(model_path(source))
fs = eeg_sampling_rate(patient)

if choix.requires_contract and contract is None:
    st.error(
        f"Le modèle `{model_path(source)}` n'a pas de fichier de contrat "
        f"(`{model_path(source).with_suffix('.json')}`). Ré-entraîne-le depuis "
        "la page **Training** pour le régénérer."
    )
    st.stop()
try:
    x_input, fs = build_model_input(patient, is_real, contract)
except (KeyError, ValueError) as exc:
    st.error(f"Les données en base ne correspondent pas au modèle : {exc}")
    st.stop()

if x_input.shape[-1] != model.cfg.input_size:
    st.error(
        f"Incompatibilité de features : le modèle attend {model.cfg.input_size} "
        f"entrées, les données en produisent {x_input.shape[-1]}."
    )
    st.stop()

x_tensor = torch.as_tensor(x_input, dtype=torch.float32)
model_version = model_path(source).stem

if choix.is_regression:
    # The head emits BDI-II points, not a probability. `predict_proba` on this
    # model raises rather than silently squashing points through a sigmoid.
    with torch.no_grad():
        delta = float(model.predict_value(x_tensor).item())
        traj = model.predict_value_sequence(x_tensor).squeeze(0).cpu().numpy().tolist()

    bdi_ref = patient.sessions[0].score_pre
    if bdi_ref is None or bdi_ref <= 0:
        st.error(
            "Impossible d'exprimer la réduction prédite : le BDI-II de référence "
            "de ce patient est absent ou nul."
        )
        st.stop()
    pct = delta / bdi_ref
    valeur = int(pct >= 0.5)
    prediction = Prediction(
        patient_id=patient.id, valeur=valeur, probabilite=float("nan"),
        date=datetime.now(), type=Prediction.REGRESSION_TYPE,
        model_version=model_version, tri_trajectory=traj,
        delta_bdi_predit=delta, reduction_predite=pct,
    )
    tri_traj = traj

    c1, c2, c3 = st.columns(3)
    c1.metric("Réduction BDI-II prédite", f"{delta:+.1f} pts",
              delta=f"{pct:.0%} du score initial")
    c2.metric("Classe dérivée (seuil 50 %)",
              "Répondeur" if valeur == 1 else "Non-répondeur")
    c3.metric(f"{unit.capitalize()}s utilisées", len(patient.sessions))
    st.caption(
        f"BDI-II de référence {bdi_ref:.0f} → score final prédit "
        f"{max(bdi_ref - delta, 0):.0f}. Le seuil de 50 % est appliqué **après** "
        f"la régression : le modèle n'a jamais vu la classe binaire."
    )
else:
    with torch.no_grad():
        proba = float(model.predict_proba(x_tensor).item())
        tri_traj = model.predict_tri(x_tensor).squeeze(0).cpu().numpy().tolist()

    valeur = int(proba >= 0.5)
    prediction = Prediction(
        patient_id=patient.id,
        valeur=valeur,
        probabilite=proba,
        date=datetime.now(),
        model_version=model_version,
        tri_trajectory=tri_traj,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilité de réponse", f"{proba:.1%}")
    c2.metric("Classe prédite", "Répondeur" if valeur == 1 else "Non-répondeur")
    c3.metric(f"{unit.capitalize()}s utilisées", len(patient.sessions))

st.write(f"**Interprétation** — {prediction.afficher()}")

if is_real:
    observed = patient.sessions[0]
    if observed.score_pre is not None and observed.score_post is not None:
        pct = (observed.score_pre - observed.score_post) / max(observed.score_pre, 1e-9)
        st.info(
            f"**Vérité terrain (BDI-II réel)** — {observed.score_pre:.0f} → "
            f"{observed.score_post:.0f}, soit {pct:.0%} de réduction → "
            f"**{'répondeur' if pct >= 0.5 else 'non-répondeur'}**."
        )

titre_traj = (
    f"##### Réduction BDI-II prédite — trajectoire par {unit}"
    if choix.is_regression
    else f"##### Indice thérapeutique (TRI) — trajectoire par {unit}"
)
st.markdown(titre_traj)
if choix.is_regression:
    st.caption(
        "Sortie du modèle après chaque époque, en **points BDI-II** — pas une "
        "probabilité, donc pas un TRI. ⚠️ L'axe est constitué d'époques d'un "
        "unique enregistrement de repos : c'est l'accumulation de preuve du "
        "modèle sur le signal, pas une évolution clinique au fil du traitement."
    )
elif is_real:
    st.caption(
        "TRIₜ = σ(W·hₜ + b). ⚠️ Sur TDBRAIN les pas de temps sont des **époques d'un "
        "unique enregistrement de repos**, pas des séances successives : cette courbe "
        "montre l'accumulation de preuve du LSTM sur le signal, **pas** une évolution "
        "clinique au fil du traitement."
    )
else:
    st.caption(
        "TRIₜ = σ(W·hₜ + b) : estimation de la probabilité de réponse au fil des séances, "
        "à mesure que le LSTM accumule les preuves (la dernière valeur = probabilité finale)."
    )
st.line_chart(
    {"ΔBDI prédit" if choix.is_regression else "TRI": tri_traj}, x=None, height=240
)

# Observed *response*, not observed severity. `analyser_ecart` expects a score in
# [0, 1] where >= 0.5 means responder, so it must be fed the BDI-II reduction
# fraction. Passing the post-treatment BDI (as `score/30`) was both the wrong
# quantity and the wrong polarity — a low post-treatment score is a *good*
# outcome, yet scored as a non-responder, so the page could show
# "60% de réduction -> répondeur" above "observe: 0, concordance: false".
first, last = patient.sessions[0], patient.sessions[-1]
pre, post = first.score_pre, last.score_post
if pre is not None and post is not None and pre > 0:
    observed_response = (pre - post) / pre
    ecart = prediction.analyser_ecart(observed_response)
    st.markdown("##### Comparaison au score clinique observé")
    st.caption(
        f"Réponse observée = réduction BDI-II ({pre:.0f} → {post:.0f}) = "
        f"{observed_response:.0%} ; seuil répondeur 50%."
    )
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
