"""Tests for seeding the database from the real-data TDBRAIN loader.

Exercised against the synthetic TDBRAIN-format tree (never real patient data).
The guarantee these protect is the one that breaks silently: the app rebuilds the
model's 130-D montage inputs *from the database*, so anything that reorders,
duplicates or drops channels between seeding and prediction corrupts the input
vector without raising.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.data.tdbrain import (
    TDBRAIN_CHANNELS_26,
    TDBRAINConfig,
    load_tdbrain,
    make_synthetic_tdbrain,
)
from src.data.tdbrain_seeder import (
    PROTOCOLS,
    montage_from_repository,
    seed_tdbrain,
)
from src.db import Repository


@pytest.fixture
def dataset(tmp_path):
    root = make_synthetic_tdbrain(tmp_path / "tdbrain", n_patients=6, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_tdbrain(
            TDBRAINConfig(root=root, n_epochs=4, epoch_seconds=1.0, target_fs=250.0)
        )


@pytest.fixture
def repo(tmp_path):
    return Repository(db_url=f"sqlite:///{tmp_path / 'seed_test.sqlite3'}")


def test_seeds_every_patient_with_full_montage(repo, dataset):
    n = seed_tdbrain(repo, dataset)
    assert n == dataset.signals_mc.shape[0]

    pid = str(dataset.metadata.iloc[0]["patient_id"])
    patient = repo.charger_patient(pid)
    assert patient is not None
    # one session per epoch, one signal per channel
    assert len(patient.sessions) == dataset.signals_mc.shape[1]
    assert len(patient.sessions[0].signaux) == len(TDBRAIN_CHANNELS_26)
    assert {s.canal for s in patient.sessions[0].signaux} == set(TDBRAIN_CHANNELS_26)
    assert patient.sessions[0].signaux[0].valeurs.shape[0] == dataset.window


def test_real_clinical_values_are_stored_not_invented(repo, dataset):
    seed_tdbrain(repo, dataset)
    row = dataset.metadata.iloc[0]
    patient = repo.charger_patient(str(row["patient_id"]))

    session = patient.sessions[0]
    assert session.score_pre == pytest.approx(float(row["bdi_pre"]))
    assert session.score_post == pytest.approx(float(row["bdi_post"]))

    # the rTMS protocol number maps onto its real published parameters
    freq, site, _ = PROTOCOLS[int(row["protocol"])]
    assert session.parametres.frequence_hz == freq
    assert session.parametres.localisation == site

    # unknown stimulation parameters stay zero rather than plausible-looking
    assert session.parametres.intensite_pct == 0.0
    assert session.parametres.nb_trains == 0


def test_scores_identical_across_epochs(repo, dataset):
    """One recording -> one treatment course: epochs must not imply a trajectory."""
    seed_tdbrain(repo, dataset)
    patient = repo.charger_patient(str(dataset.metadata.iloc[0]["patient_id"]))
    assert len({s.score_pre for s in patient.sessions}) == 1
    assert len({s.score_post for s in patient.sessions}) == 1
    assert len({s.date for s in patient.sessions}) == 1


def test_reseeding_does_not_duplicate_signals(repo, dataset):
    """Re-seeding used to stack a second copy of every channel, silently doubling
    the montage and corrupting the model's input width."""
    seed_tdbrain(repo, dataset)
    before = len(repo.charger_patient(str(dataset.metadata.iloc[0]["patient_id"])).sessions[0].signaux)
    seed_tdbrain(repo, dataset)
    after = len(repo.charger_patient(str(dataset.metadata.iloc[0]["patient_id"])).sessions[0].signaux)
    assert after == before == len(TDBRAIN_CHANNELS_26)


def test_resaving_a_patient_does_not_duplicate_signals(repo, dataset):
    """The Patients page saves a loaded patient when a note is added."""
    seed_tdbrain(repo, dataset)
    pid = str(dataset.metadata.iloc[0]["patient_id"])
    patient = repo.charger_patient(pid)
    repo.sauvegarder_patient(patient)
    assert len(repo.charger_patient(pid).sessions[0].signaux) == len(TDBRAIN_CHANNELS_26)


def test_round_trip_reproduces_the_signal(repo, dataset):
    """Channels must come back by name in montage order, not database row order."""
    seed_tdbrain(repo, dataset)
    mc, labels, groups, channels, fs, _ecg = montage_from_repository(repo)

    assert channels == list(TDBRAIN_CHANNELS_26)
    assert fs == pytest.approx(dataset.fs)
    assert mc.shape == dataset.signals_mc.shape

    order = {str(p): i for i, p in enumerate(dataset.metadata["patient_id"])}
    for row, pid in enumerate(groups):
        np.testing.assert_allclose(mc[row], dataset.signals_mc[order[pid]], rtol=1e-5)


def test_round_trip_recomputes_labels_from_stored_scores(repo, dataset):
    seed_tdbrain(repo, dataset)
    _, labels, groups, _, _, _ = montage_from_repository(repo)
    expected = {
        str(r["patient_id"]): int(r["responder"]) for _, r in dataset.metadata.iterrows()
    }
    assert [expected[pid] for pid in groups] == list(labels)


def test_seeding_rejects_a_single_channel_dataset(repo, dataset):
    dataset.signals_mc = None
    with pytest.raises(ValueError, match="montage"):
        seed_tdbrain(repo, dataset)


def test_seeding_rejects_diagnosis_task_metadata(repo, dataset):
    dataset.metadata = dataset.metadata.drop(columns=["bdi_pre", "bdi_post"])
    with pytest.raises(ValueError, match="bdi_p"):
        seed_tdbrain(repo, dataset)


def test_limit_seeds_a_subset(repo, dataset):
    assert seed_tdbrain(repo, dataset, limit=2) == 2


# --------------------------------------------------------------------------- #
# Autonomic channel round-trip.
# --------------------------------------------------------------------------- #


@pytest.fixture
def dataset_ecg(tmp_path):
    root = make_synthetic_tdbrain(
        tmp_path / "tdbrain_ecg", n_patients=6, seed=4, with_ecg=True,
        duration_seconds=30.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_tdbrain(
            TDBRAINConfig(root=root, n_epochs=4, epoch_seconds=1.0, target_fs=250.0)
        )


def test_ecg_round_trips_through_the_database(repo, dataset_ecg):
    from src.domain import SignalType

    seed_tdbrain(repo, dataset_ecg)
    mc, labels, groups, channels, fs, ecg = montage_from_repository(repo)

    # The tachogram must not leak into the EEG montage.
    assert channels == list(TDBRAIN_CHANNELS_26)
    assert mc.shape[2] == len(TDBRAIN_CHANNELS_26)
    assert fs == pytest.approx(dataset_ecg.fs)

    assert ecg is not None
    assert ecg.shape == dataset_ecg.ecg.shape
    order = {str(p): i for i, p in enumerate(dataset_ecg.metadata["patient_id"])}
    for row, pid in enumerate(groups):
        np.testing.assert_allclose(ecg[row], dataset_ecg.ecg[order[pid]], rtol=1e-5)

    # One ECG signal per session, stored with the ECG type and no sampling rate
    # (an RR series is event-sampled, so a Hz value would be meaningless).
    patient = repo.charger_patient(str(groups[0]))
    ecg_sigs = [s for s in patient.sessions[0].signaux if s.type_signal == SignalType.ECG]
    assert len(ecg_sigs) == 1
    assert ecg_sigs[0].canal == "Erbs"
    assert ecg_sigs[0].sampling_rate_hz == 0.0


def test_cohort_without_ecg_round_trips_as_none(repo, dataset):
    seed_tdbrain(repo, dataset)
    _, _, _, _, _, ecg = montage_from_repository(repo)
    assert ecg is None


def test_ecg_signal_reports_hrv_features(repo, dataset_ecg):
    """The domain object must dispatch on modality, not assume a uniform sample rate."""
    from src.domain import SignalType

    seed_tdbrain(repo, dataset_ecg)
    patient = repo.charger_patient(str(dataset_ecg.metadata.iloc[0]["patient_id"]))
    sig = next(s for s in patient.sessions[0].signaux if s.type_signal == SignalType.ECG)

    feats = sig.extraire_features()
    assert set(feats) == {
        "hrv_hr_mean", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_lf_hf",
    }
    assert 30.0 < feats["hrv_hr_mean"] < 200.0
