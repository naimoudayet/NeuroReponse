from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from src.db import Repository
from src.domain import (
    Patient,
    Prediction,
    RTMSParameters,
    SessionRTMS,
    SignalNeurophysiologique,
    SignalType,
)
from src.domain.patient import DossierClinique


@pytest.fixture
def repo(tmp_path) -> Repository:
    db_file = tmp_path / "test.sqlite3"
    return Repository(db_url=f"sqlite:///{db_file}")


def _full_patient() -> Patient:
    p = Patient(id="P001", nom="Alice", age=35, diagnostic="MDD")
    p.ajouter_dossier(DossierClinique(date=datetime(2026, 1, 1), note="intake", score_depression=22.0))

    params = RTMSParameters(
        frequence_hz=10.0,
        intensite_pct=110.0,
        duree_train_s=4.0,
        nb_trains=75,
        intervalle_train_s=26.0,
        localisation="DLPFC_gauche",
    )
    sess = SessionRTMS(id_session="S001", patient_id=p.id, parametres=params)
    sess.demarrer()
    sess.enregistrer_donnees(
        SignalNeurophysiologique(
            type_signal=SignalType.EEG,
            valeurs=np.arange(256, dtype=np.float32),
            timestamp=datetime(2026, 1, 2, 10, 0),
            canal="Fz",
            sampling_rate_hz=256.0,
        )
    )
    sess.cloturer(score_post=12.0)
    p.ajouter_session(sess)
    return p


def test_save_and_load_patient_roundtrip(repo: Repository):
    original = _full_patient()
    repo.sauvegarder_patient(original)

    loaded = repo.charger_patient(original.id)
    assert loaded is not None
    assert loaded.nom == "Alice"
    assert loaded.age == 35
    assert len(loaded.historique_clinique) == 1
    assert loaded.historique_clinique[0].score_depression == 22.0

    assert len(loaded.sessions) == 1
    sess = loaded.sessions[0]
    assert sess.id_session == "S001"
    assert sess.statut == "terminee"
    assert sess.score_post == 12.0
    assert sess.parametres.frequence_hz == 10.0
    assert sess.parametres.total_pulses() == 3000

    assert len(sess.signaux) == 1
    sig = sess.signaux[0]
    assert sig.type_signal == SignalType.EEG
    assert sig.canal == "Fz"
    np.testing.assert_array_equal(sig.valeurs, np.arange(256, dtype=np.float32))


def test_rechercher_session(repo: Repository):
    repo.sauvegarder_patient(_full_patient())
    sess = repo.rechercher_session("S001")
    assert sess is not None
    assert sess.patient_id == "P001"


def test_prediction_persistence(repo: Repository):
    repo.sauvegarder_patient(_full_patient())
    pred = Prediction(
        patient_id="P001",
        valeur=1,
        probabilite=0.78,
        date=datetime(2026, 2, 1),
        model_version="v0.1",
    )
    pred_id = repo.sauvegarder_prediction(pred)
    assert pred_id > 0

    preds = repo.lister_predictions("P001")
    assert len(preds) == 1
    assert preds[0].valeur == 1
    assert preds[0].probabilite == pytest.approx(0.78)
    assert preds[0].model_version == "v0.1"
