"""Tests for the clinical (rTMS) block rebuilt from the database.

The guarantee that matters: a patient loaded out of SQLite must produce **the same
feature row the training loader produced** for them. Two of the four clinical
features are stored indirectly — the protocol integer is recovered from the
stimulation parameters, the baseline BDI-II from the session score — so a drift
here would stay invisible while every shape assertion still passed.

The multimodal test is the strict one: it checks the full 139-column vector, which
only matches if the blocks are concatenated in the canonical ``rtms, eeg, ecg``
order *and* the EEG block alone is z-scored.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.app.inference import build_model_input, clinical_block, rebuild_tdbrain_input
from src.data.modalities import RTMS_FEATURE_NAMES, build_features
from src.data.simulator_matched import MatchedSimConfig, simulate_matched
from src.data.tdbrain_seeder import (
    _rtms_parameters,
    protocol_from_parameters,
    seed_tdbrain,
)
from src.db import Repository
from src.preprocessing.features import BANDS

N_PATIENTS = 6
N_EPOCHS = 4
WINDOW = 500


def _cohort():
    return simulate_matched(MatchedSimConfig(
        n_patients=N_PATIENTS, n_epochs=N_EPOCHS, window=WINDOW, seed=7,
    ))


def _contract(dataset, x, modalities):
    from src.models.train_tdbrain import FeatureContract

    return FeatureContract(
        source="simule",
        task="response",
        features="+".join(modalities),
        fs=float(dataset.fs),
        channels=list(dataset.channels) if "eeg" in modalities else [],
        n_bands=len(BANDS),
        per_patient_zscore=True,
        input_size=int(x.shape[-1]),
        window=int(dataset.window),
        n_epochs=int(x.shape[1]),
        modalities=list(modalities),
        ecg_channel="Erbs" if "ecg" in modalities else None,
        n_rr=int(dataset.ecg.shape[-1]) if "ecg" in modalities else 0,
    )


@pytest.fixture
def seeded(tmp_path):
    dataset = _cohort()
    repo = Repository(db_url=f"sqlite:///{tmp_path / 'clinical.sqlite3'}")
    seed_tdbrain(repo, dataset)
    return repo, dataset


def test_protocol_is_recovered_from_the_stored_parameters():
    """The seeder stores frequency + site; the block needs the arm number back."""
    for protocol in (1, 2):
        assert protocol_from_parameters(_rtms_parameters(protocol)) == protocol
    # An unknown protocol must not be guessed into a plausible one.
    assert protocol_from_parameters(_rtms_parameters(99)) is None


def test_clinical_block_matches_the_training_features(seeded):
    repo, dataset = seeded
    x, _y, _g, names = build_features(dataset, modalities=("rtms",))
    assert names == RTMS_FEATURE_NAMES

    for i, pid in enumerate(dataset.metadata["patient_id"].astype(str)):
        rebuilt = clinical_block(repo.charger_patient(pid), x.shape[1])
        np.testing.assert_allclose(rebuilt, x[i], rtol=0, atol=1e-6)


def test_age_survives_the_database_unrounded(seeded):
    """Age is the strongest single predictor here; rounding it changes the input."""
    repo, dataset = seeded
    ages = dataset.metadata["age"].astype(float).to_numpy()
    assert not np.allclose(ages, np.round(ages)), "fixture must have decimal ages"

    stored = np.array([
        repo.charger_patient(str(pid)).age
        for pid in dataset.metadata["patient_id"]
    ])
    np.testing.assert_allclose(stored, ages, rtol=0, atol=1e-6)


def test_multimodal_rebuild_matches_training_and_orders_blocks_canonically(seeded):
    repo, dataset = seeded
    modalities = ("rtms", "eeg", "ecg")
    x, _y, _g, _n = build_features(dataset, modalities=modalities)
    contract = _contract(dataset, x, modalities)

    for i, pid in enumerate(dataset.metadata["patient_id"].astype(str)):
        rebuilt = rebuild_tdbrain_input(
            repo.charger_patient(pid), contract, float(dataset.fs)
        )
        assert rebuilt.shape == (1, x.shape[1], x.shape[2])
        np.testing.assert_allclose(rebuilt[0], x[i], rtol=1e-5, atol=1e-5)


def test_missing_sexe_is_refused_rather_than_imputed(seeded):
    """Training imputes with the cohort median, which one patient cannot compute."""
    repo, dataset = seeded
    pid = str(dataset.metadata["patient_id"].iloc[0])
    patient = repo.charger_patient(pid)
    patient.sexe = None

    with pytest.raises(ValueError, match="sexe"):
        clinical_block(patient, N_EPOCHS)


def test_clinical_only_model_needs_no_recording(seeded):
    """A 4-feature model must predict on a patient carrying no signals at all."""
    repo, dataset = seeded
    x, _y, _g, _n = build_features(dataset, modalities=("rtms",))
    contract = _contract(dataset, x, ("rtms",))
    assert not contract.uses_signals

    patient = repo.charger_patient(str(dataset.metadata["patient_id"].iloc[0]))
    for session in patient.sessions:
        session.signaux.clear()

    rebuilt, fs = build_model_input(patient, is_real=True, contract=contract)
    assert rebuilt.shape == (1, contract.n_epochs, 4)
    assert fs == pytest.approx(dataset.fs)
