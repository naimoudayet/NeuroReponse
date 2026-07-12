from __future__ import annotations

from datetime import datetime

import numpy as np

from src.domain import (
    ClinicianInterface,
    Patient,
    Prediction,
    Preprocessing,
    RTMSParameters,
    SessionRTMS,
    SignalNeurophysiologique,
    SignalType,
)
from src.domain.patient import DossierClinique


def _make_patient() -> Patient:
    p = Patient(id="P001", nom="Test Patient", age=42, diagnostic="MDD")
    p.ajouter_dossier(DossierClinique(date=datetime(2026, 1, 10), note="baseline", score_depression=24.0))
    return p


def _make_session(patient_id: str) -> SessionRTMS:
    params = RTMSParameters(
        frequence_hz=10.0,
        intensite_pct=110.0,
        duree_train_s=4.0,
        nb_trains=75,
        intervalle_train_s=26.0,
        localisation="DLPFC_gauche",
        protocole="standard_depression",
    )
    sess = SessionRTMS(id_session="S001", patient_id=patient_id, parametres=params)
    sess.demarrer()
    sess.enregistrer_donnees(
        SignalNeurophysiologique(
            type_signal=SignalType.EEG,
            valeurs=np.random.RandomState(0).randn(512).astype(np.float32),
            timestamp=datetime(2026, 1, 11, 9, 30),
            canal="Fz",
            sampling_rate_hz=256.0,
        )
    )
    sess.cloturer(score_post=14.0)
    return sess


def test_patient_dossier_and_session_attachment():
    p = _make_patient()
    p.ajouter_session(_make_session(p.id))

    assert len(p.consulter_historique()) == 1
    assert p.consulter_historique()[0].score_depression == 24.0
    assert len(p.sessions) == 1
    assert p.sessions[0].statut == "terminee"


def test_rtms_parameters_total_pulses():
    params = RTMSParameters(
        frequence_hz=10.0,
        intensite_pct=110.0,
        duree_train_s=4.0,
        nb_trains=75,
        intervalle_train_s=26.0,
        localisation="DLPFC_gauche",
    )
    assert params.total_pulses() == 3000


def test_session_rapport_shape():
    sess = _make_session("P001")
    rapport = sess.generer_rapport()
    assert rapport["id_session"] == "S001"
    assert rapport["statut"] == "terminee"
    assert rapport["nb_signaux"] == 1
    assert rapport["parametres"]["total_pulses"] == 3000
    assert rapport["score_post"] == 14.0


def test_preprocessing_normalize_zero_mean_unit_var():
    pre = Preprocessing()
    x = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    z = pre.normaliser(x)
    assert np.allclose(z.mean(axis=-1), 0.0, atol=1e-7)
    assert np.allclose(z.std(axis=-1), 1.0, atol=1e-7)


def test_prediction_afficher_and_ecart():
    pred = Prediction(
        patient_id="P001",
        valeur=1,
        probabilite=0.82,
        date=datetime(2026, 2, 1),
    )
    assert "Répondeur" in pred.afficher()
    assert "82" in pred.afficher()

    ecart = pred.analyser_ecart(score_clinique_observe=0.9)
    assert ecart["concordance"] is True
    assert ecart["predit"] == 1


def test_clinician_modifier_parametres():
    ci = ClinicianInterface(nom_utilisateur="dr_test")
    params = RTMSParameters(
        frequence_hz=10.0,
        intensite_pct=110.0,
        duree_train_s=4.0,
        nb_trains=75,
        intervalle_train_s=26.0,
        localisation="DLPFC_gauche",
    )
    updated = ci.modifier_parametres(params, frequence_hz=20.0, intensite_pct=120.0)
    assert updated.frequence_hz == 20.0
    assert updated.intensite_pct == 120.0
