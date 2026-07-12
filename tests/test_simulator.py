from __future__ import annotations

import numpy as np
from scipy.signal import welch

from src.data.simulator import SimConfig, simulate, save
from src.data.loader import load


def test_shape_and_labels():
    ds = simulate(SimConfig(n_patients=40, n_sessions=8, window=128, seed=0))
    assert ds.signals.shape == (40, 8, 128)
    assert ds.signals.dtype == np.float32
    assert ds.labels.shape == (40,)
    assert set(np.unique(ds.labels)).issubset({0, 1})


def test_label_balance_close_to_target():
    ds = simulate(SimConfig(n_patients=200, responder_rate=0.5, seed=1))
    rate = ds.labels.mean()
    assert 0.45 <= rate <= 0.55


def test_metadata_rows_match_patients_x_sessions():
    cfg = SimConfig(n_patients=10, n_sessions=5, seed=2)
    ds = simulate(cfg)
    assert len(ds.metadata) == cfg.n_patients * cfg.n_sessions
    assert set(ds.metadata.columns) >= {
        "patient_id", "session_idx", "label", "alpha_gain",
        "score_clinique", "frequence_hz", "intensite_pct", "localisation",
    }


def test_seed_reproducibility():
    a = simulate(SimConfig(n_patients=15, seed=7))
    b = simulate(SimConfig(n_patients=15, seed=7))
    np.testing.assert_array_equal(a.signals, b.signals)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_responders_have_higher_late_alpha():
    """The simulator must inject a learnable signal — otherwise Phase 4 is moot."""
    cfg = SimConfig(n_patients=120, n_sessions=10, seed=3)
    ds = simulate(cfg)

    last = ds.signals[:, -1, :]  # last session per patient
    freqs, psd = welch(last, fs=cfg.fs, nperseg=cfg.window, axis=-1)
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    alpha_power = psd[:, alpha_mask].mean(axis=-1)

    resp_alpha = alpha_power[ds.labels == 1].mean()
    nonresp_alpha = alpha_power[ds.labels == 0].mean()
    assert resp_alpha > nonresp_alpha, (
        f"Responders should have higher late-session alpha power "
        f"(got {resp_alpha:.3f} vs {nonresp_alpha:.3f})"
    )


def test_save_and_load_roundtrip(tmp_path):
    ds = simulate(SimConfig(n_patients=5, n_sessions=3, window=64, seed=11))
    save(ds, tmp_path)
    loaded = load(tmp_path)
    np.testing.assert_array_equal(loaded.signals, ds.signals)
    np.testing.assert_array_equal(loaded.labels, ds.labels)
    assert loaded.fs == ds.config.fs
    assert loaded.window == ds.config.window
    assert len(loaded.metadata) == 5 * 3


def test_seeder_persists_patients(tmp_path):
    from src.data.seeder import seed
    from src.db import Repository

    ds = simulate(SimConfig(n_patients=4, n_sessions=3, window=64, seed=5))
    save(ds, tmp_path)
    loaded = load(tmp_path)

    db_path = tmp_path / "seed.sqlite3"
    repo = Repository(db_url=f"sqlite:///{db_path}")
    n = seed(repo, dataset=loaded)
    assert n == 4

    p0 = repo.charger_patient("P000")
    assert p0 is not None
    assert len(p0.sessions) == 3
    assert p0.sessions[0].signaux[0].valeurs.shape == (64,)
