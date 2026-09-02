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


# --------------------------------------------------------------------------- #
# Session ordering. The LSTM's sequence axis *is* `patient.sessions`, so the
# order this comes back in decides what the clinical loop predicts on.
# --------------------------------------------------------------------------- #
def _bare_session(pid: str, index: int, date: datetime) -> SessionRTMS:
    params = RTMSParameters(
        frequence_hz=10.0, intensite_pct=110.0, duree_train_s=4.0,
        nb_trains=75, intervalle_train_s=26.0, localisation="DLPFC_gauche",
    )
    sess = SessionRTMS(
        id_session=f"{pid}-S{index:02d}", patient_id=pid,
        parametres=params, date=date,
    )
    sess.enregistrer_donnees(
        SignalNeurophysiologique(
            type_signal=SignalType.EEG,
            valeurs=np.arange(128, dtype=np.float32),
            timestamp=date, canal="Fz", sampling_rate_hz=256.0,
        )
    )
    sess.score_pre = 30.0
    sess.cloturer(score_post=30.0 - index)
    return sess


def test_sessions_come_back_in_chronological_order(repo: Repository):
    """A visit recorded late must not reorder the treatment course.

    Without an explicit ORDER BY, SQLite hands back rowid (= insertion) order.
    A clinician backfilling a missed session would then feed the LSTM a shuffled
    sequence and get a wrong TRI with no error anywhere — this pins the order to
    the clinical timeline instead.
    """
    repo.sauvegarder_patient(Patient(id="P7", nom="Bob", age=40.0, diagnostic="MDD"))

    # Recorded out of order: session 2, then the missed session 1, then 3.
    for index, day in [(2, 3), (1, 1), (3, 5)]:
        p = repo.charger_patient("P7")
        p.ajouter_session(_bare_session("P7", index, datetime(2026, 1, day)))
        repo.sauvegarder_patient(p)

    p = repo.charger_patient("P7")
    assert [s.id_session for s in p.sessions] == ["P7-S01", "P7-S02", "P7-S03"]
    assert [s.date for s in p.sessions] == sorted(s.date for s in p.sessions)

    # The other read path must agree, or the two disagree about the same course.
    assert [s.id_session for s in repo.lister_sessions_patient("P7")] == [
        "P7-S01", "P7-S02", "P7-S03",
    ]


def test_epoch_sessions_sharing_a_timestamp_keep_their_id_order(repo: Repository):
    """The research cohorts stamp every epoch with one reference date.

    Ordering on the date alone would leave those tied and back to rowid order,
    so the id is the tiebreaker.
    """
    repo.sauvegarder_patient(Patient(id="P8", nom="Eve", age=50.0, diagnostic="MDD"))
    meme_date = datetime(2020, 1, 1)
    p = repo.charger_patient("P8")
    for index in [3, 1, 2]:
        p.ajouter_session(_bare_session("P8", index, meme_date))
    repo.sauvegarder_patient(p)

    p = repo.charger_patient("P8")
    assert [s.id_session for s in p.sessions] == ["P8-S01", "P8-S02", "P8-S03"]


def test_demarrer_does_not_overwrite_a_supplied_date():
    """Replaying a historical course must keep its own dates.

    `demarrer()` stamping `now()` unconditionally collapsed the seeder's 10
    sessions — spread over 20 days — onto a single millisecond, erasing the
    trajectory the follow-up page exists to plot.
    """
    params = RTMSParameters(
        frequence_hz=10.0, intensite_pct=110.0, duree_train_s=4.0,
        nb_trains=75, intervalle_train_s=26.0, localisation="DLPFC_gauche",
    )
    voulue = datetime(2026, 3, 14, 9, 30)
    sess = SessionRTMS(id_session="S1", patient_id="P1", parametres=params, date=voulue)
    sess.demarrer(date=voulue)
    assert sess.statut == "en_cours"
    assert sess.date == voulue


def test_remplacer_sessions_drops_ids_the_new_course_no_longer_has(repo: Repository):
    """Re-seeding must yield exactly the new cohort, not the union of both runs.

    `sauvegarder_patient` upserts and never deletes — correct for the clinical
    loop, wrong for a seeder: a renamed session id would otherwise survive from
    the previous run and be fed to the model as an extra visit.
    """
    repo.sauvegarder_patient(Patient(id="P9", nom="Zoe", age=44.0, diagnostic="MDD"))

    ancien = repo.charger_patient("P9")
    for index in (1, 2, 3):
        ancien.ajouter_session(_bare_session("P9", index, datetime(2026, 1, index)))
    repo.remplacer_sessions_patient(ancien)
    assert len(repo.charger_patient("P9").sessions) == 3

    nouveau = Patient(id="P9", nom="Zoe", age=44.0, diagnostic="MDD")
    nouveau.ajouter_session(_bare_session("P9", 7, datetime(2026, 2, 1)))
    repo.remplacer_sessions_patient(nouveau)

    apres = repo.charger_patient("P9")
    assert [s.id_session for s in apres.sessions] == ["P9-S07"]
    # The orphaned sessions' signals must go too, or they accumulate forever.
    assert repo.rechercher_session("P9-S01") is None


def test_sauvegarder_patient_still_only_appends(repo: Repository):
    """The clinical loop's contract: recording session n+1 keeps 1..n."""
    repo.sauvegarder_patient(Patient(id="PA", nom="Ann", age=44.0, diagnostic="MDD"))
    for index in (1, 2):
        p = repo.charger_patient("PA")
        p.ajouter_session(_bare_session("PA", index, datetime(2026, 1, index)))
        repo.sauvegarder_patient(p)

    seule = Patient(id="PA", nom="Ann", age=44.0, diagnostic="MDD")
    seule.ajouter_session(_bare_session("PA", 3, datetime(2026, 1, 3)))
    repo.sauvegarder_patient(seule)

    assert [s.id_session for s in repo.charger_patient("PA").sessions] == [
        "PA-S01", "PA-S02", "PA-S03",
    ]
