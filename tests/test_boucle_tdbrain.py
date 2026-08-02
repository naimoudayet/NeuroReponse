"""Tests for the TDBRAIN arm of the clinical loop (per-session snapshot).

The guarantee that matters: a loop session must produce **the same feature vector
the training loader would have produced** for that recording. If the two paths
drift, the model is trained on one preprocessing and used on another, and nothing
downstream would reveal it.
"""
from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pytest

from src.app.inference import snapshot_input
from src.data.tdbrain import (
    TDBRAIN_CHANNELS_26,
    TDBRAINConfig,
    load_tdbrain,
    make_synthetic_tdbrain,
    tdbrain_features,
)
from src.domain import RTMSParameters, SignalType
from src.models.train_tdbrain import FeatureContract
from src.reporting.boucle import construire_session_montage
from src.reporting.enregistrement import duree_secondes, lire_enregistrement

PARAMS = RTMSParameters(
    frequence_hz=10.0, intensite_pct=0.0, duree_train_s=0.0, nb_trains=0,
    intervalle_train_s=0.0, localisation="L-DLPFC", protocole="saisie clinique",
)


def _contract(n_epochs=4, window=250, fs=250.0, modalities=("eeg", "ecg")):
    return FeatureContract(
        source="tdbrain", task="response", features="montage_band_powers+hrv",
        fs=fs, channels=list(TDBRAIN_CHANNELS_26), n_bands=5,
        per_patient_zscore=False,
        input_size=len(TDBRAIN_CHANNELS_26) * 5 + (5 if "ecg" in modalities else 0),
        window=window, n_epochs=n_epochs, modalities=list(modalities),
        ecg_channel="Erbs", n_rr=64,
    )


@pytest.fixture
def synth_root(tmp_path):
    return make_synthetic_tdbrain(
        tmp_path / "clin", n_patients=4, seed=11, with_ecg=True, duration_seconds=30.0
    )


def _session_from(montage, tach, fs, n=1):
    return construire_session_montage(
        patient_id="PC1", index=n, parametres=PARAMS, montage=montage, fs=fs,
        score_pre=30.0, score_post=18.0, tachogram=tach,
        date=datetime(2026, 3, 1),
    )


def test_uploaded_recording_parses_into_montage_and_tachogram(synth_root):
    rec = sorted(synth_root.rglob("*restEO*.csv"))[0]
    montage, tach, fs = lire_enregistrement(rec, TDBRAIN_CHANNELS_26, fs_cible=250.0)

    assert set(montage) == set(TDBRAIN_CHANNELS_26)
    assert len({v.size for v in montage.values()}) == 1        # rectangular
    assert fs == 250.0
    assert tach is not None and tach.size >= 3
    assert np.all(tach > 0.3) and np.all(tach < 2.0)           # physiological RR
    assert duree_secondes(montage, fs) == pytest.approx(30.0, abs=0.5)


def test_session_stores_the_whole_recording_not_one_epoch(synth_root):
    rec = sorted(synth_root.rglob("*restEO*.csv"))[0]
    montage, tach, fs = lire_enregistrement(rec, TDBRAIN_CHANNELS_26, fs_cible=250.0)
    sess = _session_from(montage, tach, fs)

    eeg = [s for s in sess.signaux if s.type_signal == SignalType.EEG]
    ecg = [s for s in sess.signaux if s.type_signal == SignalType.ECG]
    assert len(eeg) == len(TDBRAIN_CHANNELS_26)
    assert len(ecg) == 1
    assert ecg[0].sampling_rate_hz == 0.0                      # event-sampled
    # One clinical session = one full recording, not a single epoch window.
    assert eeg[0].valeurs.size == montage["Fp1"].size
    assert sess.id_session == "PC1-S01"


def test_snapshot_matches_what_the_training_loader_would_produce(synth_root):
    """The whole point: loop features == loader features for the same recording."""
    cfg = TDBRAINConfig(root=synth_root, n_epochs=4, epoch_seconds=1.0, target_fs=250.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = load_tdbrain(cfg)
    x_loader, _, _, _ = tdbrain_features(
        ds, per_patient_zscore=False, modalities=("eeg", "ecg")
    )

    pid = str(ds.metadata.iloc[0]["patient_id"])
    rec = sorted(synth_root.rglob(f"*{pid}*restEO*.csv"))[0]
    montage, tach, fs = lire_enregistrement(rec, TDBRAIN_CHANNELS_26, fs_cible=250.0)
    sess = _session_from(montage, tach, fs)

    x_loop = snapshot_input(sess, _contract(n_epochs=4, window=250))

    assert x_loop.shape == (1, 4, x_loader.shape[-1])
    np.testing.assert_allclose(x_loop[0], x_loader[0], rtol=1e-4, atol=1e-4)


def test_snapshot_refuses_a_recording_that_is_too_short(synth_root):
    rec = sorted(synth_root.rglob("*restEO*.csv"))[0]
    montage, tach, fs = lire_enregistrement(rec, TDBRAIN_CHANNELS_26, fs_cible=250.0)
    sess = _session_from(montage, tach, fs)

    # 8 x 2000 = 16000 samples needed; the fixture holds 7500.
    with pytest.raises(ValueError, match="trop court"):
        snapshot_input(sess, _contract(n_epochs=8, window=2000))


def test_snapshot_refuses_a_missing_channel(synth_root):
    rec = sorted(synth_root.rglob("*restEO*.csv"))[0]
    montage, tach, fs = lire_enregistrement(rec, TDBRAIN_CHANNELS_26, fs_cible=250.0)
    montage.pop("Pz")
    sess = _session_from(montage, tach, fs)

    with pytest.raises(ValueError, match="canaux absents"):
        snapshot_input(sess, _contract(n_epochs=4, window=250))


def test_snapshot_refuses_when_ecg_is_required_but_absent(synth_root):
    rec = sorted(synth_root.rglob("*restEO*.csv"))[0]
    montage, _tach, fs = lire_enregistrement(rec, TDBRAIN_CHANNELS_26, fs_cible=250.0)
    sess = _session_from(montage, None, fs)

    with pytest.raises(ValueError, match="tachogramme"):
        snapshot_input(sess, _contract(n_epochs=4, window=250))


def test_eeg_only_contract_works_without_ecg(synth_root):
    rec = sorted(synth_root.rglob("*restEO*.csv"))[0]
    montage, _t, fs = lire_enregistrement(rec, TDBRAIN_CHANNELS_26, fs_cible=250.0)
    sess = _session_from(montage, None, fs)

    x = snapshot_input(sess, _contract(n_epochs=4, window=250, modalities=("eeg",)))
    assert x.shape == (1, 4, len(TDBRAIN_CHANNELS_26) * 5)
    assert np.isfinite(x).all()


def test_montage_with_ragged_channels_is_rejected():
    montage = {c: np.zeros(100) for c in TDBRAIN_CHANNELS_26}
    montage["Pz"] = np.zeros(50)
    with pytest.raises(ValueError, match="même longueur"):
        construire_session_montage("P1", 1, PARAMS, montage, 250.0, 30.0, 20.0)


def test_montage_with_non_finite_values_is_rejected():
    montage = {c: np.zeros(100) for c in TDBRAIN_CHANNELS_26}
    montage["Cz"][3] = np.nan
    with pytest.raises(ValueError, match="non finies"):
        construire_session_montage("P1", 1, PARAMS, montage, 250.0, 30.0, 20.0)


def test_empty_montage_is_rejected():
    with pytest.raises(ValueError, match="vide"):
        construire_session_montage("P1", 1, PARAMS, {}, 250.0, 30.0, 20.0)
