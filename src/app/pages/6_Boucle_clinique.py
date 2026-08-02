"""Closed clinical loop: record a session → predict → adjust the stimulator → repeat.

Both cohorts support the loop, but they accumulate evidence differently, because
the two models were trained on different sequence axes:

* **Simulated** — the model consumes a sequence of *treatment sessions*, so each
  new session lengthens its input and the prediction accumulates **inside** the
  LSTM. ``predict_tri()[k]`` is the estimate after session ``k+1``.
* **TDBRAIN** — the model is *baseline-only*: it was trained on ``n_epochs``
  windows of a **single** resting recording. So every loop iteration re-records
  and predicts on that recording **alone**, giving one independent probability per
  session. The accumulation is clinical (a trend across sessions), never inside
  the recurrent axis — feeding sessions weeks apart as timesteps would be a
  distribution the model has never seen, and the contract's ``n_epochs`` check
  refuses it outright.

Adjusting the stimulator happens on the machine, outside this application. What
the page contributes is the record of which settings produced which prediction.
"""
from __future__ import annotations

import io
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from src.app.inference import build_model_input, snapshot_input
from src.app.utils import (
    get_repository,
    has_trained_model,
    list_patient_ids,
    model_choice,
    model_path,
    source_config,
    source_selector,
)
from src.domain import Patient, RTMSParameters
from src.models.train import load_model
from src.models.train_tdbrain import load_contract
from src.reporting.boucle import (
    SEUIL_REPONSE,
    construire_session,
    construire_session_montage,
    etapes_boucle,
    prochain_index,
    recommandation,
)
from src.reporting.enregistrement import (
    duree_secondes,
    generer_enregistrement_demo,
    lire_enregistrement,
)

st.set_page_config(page_title="Boucle clinique", page_icon=":repeat:", layout="wide")

source = source_selector()
cfg_src = source_config(source)
choix = model_choice(source)
repo = get_repository(source)
is_real = cfg_src.is_real
# Snapshot mode is a property of the *cohort*, not of it being real: the matched
# simulated cohort is baseline-only too, so it accumulates clinically rather than
# inside the LSTM. Only the legacy sequential cohort feeds a growing sequence.
snapshot = not cfg_src.sequentiel

st.title("Boucle clinique — séance par séance")
st.caption(
    "Enregistre une séance → le modèle prédit → tu ajustes le stimulateur "
    "(hors application) → séance suivante, et ainsi de suite jusqu'à obtenir une "
    "réponse satisfaisante."
)

if not has_trained_model(source):
    st.error("Aucun modèle entraîné pour cette source — va sur la page **Training**.")
    st.stop()


@st.cache_resource
def _model(path_str: str):
    return load_model(path_str)


model = _model(str(model_path(source)))
contract = load_contract(model_path(source)) if choix.requires_contract else None

if choix.requires_contract and contract is None:
    st.error(
        f"Le checkpoint `{model_path(source)}` n'a pas de fichier de contrat — "
        "ré-entraîne-le depuis la page **Training**."
    )
    st.stop()

if not choix.uses_signals:
    st.warning(
        "**Ce modèle ne lit aucun enregistrement.** Il n'utilise que le protocole "
        "rTMS, l'âge, le sexe et le BDI-II de référence — des variables qui ne "
        "changent pas d'une séance à l'autre. La boucle rendrait donc exactement la "
        "même probabilité à chaque itération, ce qui ressemblerait à une courbe "
        "plate alors qu'aucune mesure nouvelle n'est entrée dans le modèle. "
        "Sélectionne le modèle **multimodal** dans la barre latérale pour suivre "
        "une évolution séance par séance."
    )
    st.stop()

if snapshot:
    besoin_s = contract.window * contract.n_epochs / contract.fs
    st.info(
        f"**Mode instantané ({cfg_src.label}).** Le modèle est *baseline-only* : à "
        f"chaque séance tu enregistres un **nouvel EEG de repos** "
        f"({len(contract.channels)} canaux + ECG, ≥ {besoin_s:.0f} s à "
        f"{contract.fs:g} Hz), et il prédit sur **cet enregistrement seul**. La "
        f"boucle suit ensuite l'évolution de ces prédictions successives — "
        f"l'accumulation est clinique, pas interne au LSTM."
    )
else:
    st.info(
        "**Mode séquentiel (simulé).** Le modèle consomme la suite des séances : "
        "chaque nouvelle séance allonge son entrée et la prédiction accumule "
        "l'historique **à l'intérieur** du LSTM."
    )

# --------------------------------------------------------------------------- #
# Patient selection.
# --------------------------------------------------------------------------- #
patient_ids = list_patient_ids(repo)
col_sel, col_new = st.columns([3, 2])
with col_sel:
    pid = st.selectbox("Patient", patient_ids) if patient_ids else None
with col_new:
    with st.popover("➕ Nouveau patient"):
        with st.form("nouveau_patient", clear_on_submit=True):
            n_id = st.text_input("ID", placeholder="P900")
            n_nom = st.text_input("Nom")
            n_age = st.number_input("Âge", 0.0, 120.0, 45.0, 0.5)
            # The clinical block reads sexe; a patient created without it cannot
            # be predicted on by any of the four comparison models.
            n_sexe = st.selectbox(
                "Sexe", [0, 1],
                format_func=lambda v: f"{v} — {'femme' if v == 0 else 'homme'}",
                help="Codé 0/1 comme dans la table source, tel que le modèle l'a vu.",
            )
            if st.form_submit_button("Créer") and n_id and n_nom:
                repo.sauvegarder_patient(
                    Patient(id=n_id, nom=n_nom, age=float(n_age),
                            sexe=int(n_sexe), diagnostic="MDD")
                )
                # toast, not success: the rerun below repaints the page immediately
                # and would wipe an st.success before the user ever sees it.
                st.toast(f"Patient {n_id} créé.", icon="✅")
                st.rerun()

if not pid:
    st.info("Aucun patient. Crée-en un pour démarrer une boucle.")
    st.stop()

patient = repo.charger_patient(pid)

# The seeded TDBRAIN cohort stores one *epoch* per session (ids like `-E00`), which
# is not a treatment visit and is far too short for a snapshot. Say so precisely
# instead of letting snapshot_input raise a generic "recording too short".
est_cohorte_recherche = snapshot and any(
    "-E" in s.id_session for s in patient.sessions
)
if est_cohorte_recherche:
    st.warning(
        f"**{pid} appartient à la cohorte de recherche ({cfg_src.label}).** Ses « séances » "
        f"sont les {len(patient.sessions)} époques d'un unique enregistrement de "
        f"repos avant traitement — ce n'est pas un suivi de traitement, et chaque "
        f"époque est trop courte pour une prédiction instantanée. Crée un "
        f"**nouveau patient** pour démarrer une boucle clinique."
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Predictions: accumulating (simulated) vs per-session snapshot (TDBRAIN).
# --------------------------------------------------------------------------- #
def _predictions(p) -> list[float] | None:
    """One probability per session, however this source accumulates evidence."""
    if not p.sessions or not all(s.signaux for s in p.sessions):
        return None
    try:
        if snapshot:
            out = []
            for sess in p.sessions:
                # The patient carries the clinical block; the session carries the
                # recording. A multimodal variant needs both.
                x = snapshot_input(sess, contract, patient=p)
                with torch.no_grad():
                    out.append(float(model.predict_proba(
                        torch.as_tensor(x, dtype=torch.float32)).item()))
            return out
        x, _fs = build_model_input(p, is_real=False)
        with torch.no_grad():
            return model.predict_tri(
                torch.as_tensor(x, dtype=torch.float32)
            ).squeeze(0).cpu().numpy().tolist()
    except (ValueError, KeyError, RuntimeError, IndexError) as exc:
        st.warning(f"Prédiction indisponible : {exc}")
        return None


tri = _predictions(patient)
etapes = etapes_boucle(patient, tri)
prochain = prochain_index(patient)

c1, c2, c3 = st.columns(3)
c1.metric("Séances enregistrées", len(patient.sessions))
courant = etapes[-1].tri if etapes and etapes[-1].tri is not None else None
c2.metric("P(réponse) actuelle", f"{courant:.0%}" if courant is not None else "—")
dlt = etapes[-1].delta_tri if etapes and etapes[-1].delta_tri is not None else None
c3.metric("Δ depuis la séance précédente", f"{dlt:+.0%}" if dlt is not None else "—",
          delta=f"{dlt:+.0%}" if dlt is not None else None)

# --------------------------------------------------------------------------- #
# Record the next session.
# --------------------------------------------------------------------------- #
st.markdown(f"##### Enregistrer la séance {prochain}")

d = patient.sessions[-1].parametres if patient.sessions else None
base_score = patient.sessions[0].score_pre if patient.sessions else 30.0

with st.form("nouvelle_seance"):
    st.markdown("**Paramètres de stimulation appliqués sur la machine**")
    st.caption(
        "Pré-remplis avec ceux de la séance précédente — modifie ce que tu as "
        "changé sur le stimulateur, l'application enregistre le réglage utilisé."
    )
    p1, p2, p3 = st.columns(3)
    freq = p1.number_input("Fréquence (Hz)", 0.1, 100.0, float(d.frequence_hz) if d else 10.0, 0.5)
    inten = p2.number_input("Intensité (% SM)", 0.0, 200.0, float(d.intensite_pct) if d else 110.0, 5.0)
    loc = p3.text_input("Localisation", d.localisation if d else
                        ("L-DLPFC" if snapshot else "DLPFC_gauche"))
    p4, p5, p6 = st.columns(3)
    dur = p4.number_input("Durée train (s)", 0.0, 60.0, float(d.duree_train_s) if d else 4.0, 0.5)
    nbt = p5.number_input("Nb trains", 0, 500, int(d.nb_trains) if d else 75, 5)
    iti = p6.number_input("Intervalle train (s)", 0.0, 120.0, float(d.intervalle_train_s) if d else 26.0, 1.0)

    st.markdown("**Enregistrement EEG de la séance**")
    if snapshot:
        st.caption(
            f"Fichier **BDF** ou **CSV à en-têtes de canaux** contenant les "
            f"{len(contract.channels)} canaux du montage + la dérivation "
            f"`{contract.ecg_channel or '—'}`. Le fichier passe par le même "
            f"prétraitement que les données d'entraînement (notch 50 Hz, "
            f"passe-bande 1–45 Hz, rééchantillonnage {contract.fs:g} Hz)."
        )
        types = ["bdf", "csv"]
    else:
        st.caption("Fichier CSV ou NPY : une colonne de valeurs pour le canal EEG.")
        types = ["csv", "npy"]

    origine = st.radio("Origine du signal", ["Téléverser un fichier", "Générer (démonstration)"],
                       horizontal=True,
                       help="En usage réel le signal vient de l'amplificateur ; la "
                            "génération sert à dérouler la boucle sans matériel.")
    up = st.file_uploader("Fichier", type=types)
    fs_in = st.number_input(
        "Fréquence d'échantillonnage (Hz)", 1.0, 5000.0,
        float(contract.fs) if snapshot else 256.0, 1.0,
        disabled=snapshot,
        help="Imposée par le contrat du modèle." if snapshot else None,
    )

    st.markdown("**Score clinique**")
    s1, s2 = st.columns(2)
    score_pre = s1.number_input("BDI-II de référence", 0.0, 63.0, float(base_score or 30.0), 0.5)
    score_post = s2.number_input("BDI-II après cette séance", 0.0, 63.0, float(base_score or 30.0), 0.5)

    submitted = st.form_submit_button("Enregistrer la séance et re-prédire", type="primary")

if submitted:
    params = RTMSParameters(
        frequence_hz=float(freq), intensite_pct=float(inten),
        duree_train_s=float(dur), nb_trains=int(nbt),
        intervalle_train_s=float(iti), localisation=loc,
        protocole="saisie clinique",
    )
    erreur = None
    sess = None
    try:
        if snapshot:
            besoin = contract.window * contract.n_epochs
            if origine.startswith("Téléverser"):
                if up is None:
                    raise ValueError("aucun fichier téléversé.")
                # The TDBRAIN readers work on paths (mne opens BDF by filename), so
                # the upload is spooled to a temp file and removed straight after.
                suffix = Path(up.name).suffix.lower() or ".csv"
                tmp = Path(tempfile.mkdtemp()) / f"seance{suffix}"
                tmp.write_bytes(up.getvalue())
                try:
                    montage, tach, fsr = lire_enregistrement(
                        tmp, contract.channels, fs_cible=float(contract.fs),
                        ecg_channel=contract.ecg_channel, n_rr=contract.n_rr or 64,
                    )
                finally:
                    tmp.unlink(missing_ok=True)
            else:
                montage, tach, fsr = generer_enregistrement_demo(
                    contract.channels, float(contract.fs),
                    duree_s=besoin / contract.fs + 2.0, seed=prochain,
                    n_rr=contract.n_rr or 64,
                    alpha=1.2 + 0.15 * prochain, bpm=72.0 - 1.5 * prochain,
                )
            n_samp = duree_secondes(montage, fsr) * fsr
            if n_samp < besoin:
                raise ValueError(
                    f"enregistrement trop court : {n_samp:.0f} échantillons, "
                    f"il en faut {besoin} ({besoin / contract.fs:.0f} s)."
                )
            if "ecg" in contract.modalities and tach is None:
                raise ValueError(
                    "le modèle attend la modalité ECG mais aucun pic R exploitable "
                    f"n'a été détecté sur la dérivation « {contract.ecg_channel} »."
                )
            sess = construire_session_montage(
                patient_id=pid, index=prochain, parametres=params, montage=montage,
                fs=fsr, score_pre=float(score_pre), score_post=float(score_post),
                tachogram=tach, ecg_canal=contract.ecg_channel or "Erbs",
                date=datetime.now(),
            )
        else:
            window = (patient.sessions[0].signaux[0].valeurs.shape[0]
                      if patient.sessions and patient.sessions[0].signaux else 128)
            if origine.startswith("Téléverser"):
                if up is None:
                    raise ValueError("aucun fichier téléversé.")
                raw = up.getvalue()
                if up.name.lower().endswith(".npy"):
                    signal = np.load(io.BytesIO(raw), allow_pickle=False).ravel()
                else:
                    signal = np.loadtxt(io.StringIO(raw.decode("utf-8")), delimiter=",").ravel()
            else:
                rng = np.random.default_rng(prochain)
                t = np.arange(window) / fs_in
                signal = np.sin(2 * np.pi * 10.0 * t) + 0.5 * rng.standard_normal(window)
            if signal.size < window:
                raise ValueError(
                    f"le signal doit contenir au moins {window} échantillons "
                    f"(reçu {signal.size})."
                )
            sess = construire_session(
                patient_id=pid, index=prochain, parametres=params,
                signal=signal[:window], fs=float(fs_in),
                score_pre=float(score_pre), score_post=float(score_post),
                date=datetime.now(),
            )
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        erreur = str(exc)

    if erreur:
        st.error(f"Séance non enregistrée : {erreur}")
    else:
        patient.ajouter_session(sess)
        repo.sauvegarder_patient(patient)
        st.toast(f"Séance {prochain} enregistrée.", icon="✅")
        st.rerun()

st.divider()

# --------------------------------------------------------------------------- #
# The loop's history.
# --------------------------------------------------------------------------- #
st.markdown("##### Évolution de la prédiction au fil des séances")
if not etapes:
    st.info("Aucune séance : enregistre la première pour démarrer la boucle.")
else:
    vals = [e.tri for e in etapes if e.tri is not None]
    if vals:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(vals) + 1)), y=vals, mode="lines+markers",
            line=dict(color="#a855f7", width=3), name="P(réponse)",
        ))
        fig.add_hline(y=SEUIL_REPONSE, line_dash="dash", line_color="#22c55e",
                      annotation_text="Seuil répondeur 50%")
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(
            xaxis_title="Séance", yaxis_title="P(réponse)", height=320,
            margin=dict(l=40, r=20, t=20, b=40), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Chaque point est une prédiction **indépendante** sur l'enregistrement "
            "de cette séance ; la tendance se lit d'un point à l'autre."
            if snapshot else
            "Chaque point utilise **toutes les séances jusqu'à celle-ci** : le point "
            "n°k est la prédiction du modèle après k séances."
        )

    st.markdown("##### Journal de la boucle")
    st.dataframe([e.to_row() for e in etapes], hide_index=True, use_container_width=True)

    st.markdown("##### Recommandation pour la prochaine séance")
    for msg in recommandation(etapes):
        if msg.startswith("⚠️"):
            st.warning(msg)
        elif msg.startswith("✅"):
            st.success(msg)
        else:
            st.markdown(f"- {msg}")
    st.caption(
        "La recommandation indique la **direction** observée, pas un réglage précis : "
        "aucune donnée de ce projet ne relie une intensité de stimulation à la réponse "
        "(TDBRAIN ne publie même pas la dose). Le choix des paramètres reste clinique."
    )
    if snapshot:
        st.warning(
            "⚠️ **Validité du modèle.** Sur cette cohorte la prédiction de réponse "
            "est **au niveau du hasard** (AUC 0.49–0.57, exactitude = taux de base). "
            "Cette boucle démontre le *workflow* ; elle ne doit pas guider une "
            "décision thérapeutique tant qu'un modèle discriminant n'est pas "
            "disponible."
        )
