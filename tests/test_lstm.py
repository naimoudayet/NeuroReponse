from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.simulator import SimConfig, simulate
from src.domain import ModeleLSTM
from src.models.lstm import LSTMConfig, ResponseLSTM
from src.models.train import (
    TrainConfig,
    cross_validate,
    fit_final_model,
    load_model,
    save_model,
)
from src.preprocessing.pipeline import PipelineConfig, preprocess


def _preprocessed_toy(seed: int = 0):
    ds = simulate(SimConfig(n_patients=40, n_sessions=10, window=128, seed=seed))
    out = preprocess(ds.signals, PipelineConfig(fs=256.0, mode="features"))
    groups = np.arange(out.x.shape[0])
    return out.x, ds.labels.astype(np.float32), groups


def test_forward_pass_shape():
    cfg = LSTMConfig(input_size=8)
    model = ResponseLSTM(cfg)
    x = torch.randn(4, 10, 8)
    logits = model(x)
    assert logits.shape == (4,)


def test_forward_rejects_wrong_ndim():
    model = ResponseLSTM(LSTMConfig(input_size=8))
    with pytest.raises(ValueError):
        model(torch.zeros(4, 8))


def test_cross_validate_learns_on_simulated_data():
    """Sanity check: with the engineered signal in the simulator, CV should clear chance."""
    x, y, groups = _preprocessed_toy(seed=0)
    cv = cross_validate(
        x, y, groups,
        train_cfg=TrainConfig(epochs=12, batch_size=8, lr=5e-3, early_stopping_patience=4, seed=0),
        n_splits=4,
    )
    summary = cv.summary()
    assert summary["accuracy_mean"] >= 0.55, summary
    assert summary["auc_mean"] >= 0.6, summary


def test_groupkfold_keeps_patients_separated():
    x, y, groups = _preprocessed_toy(seed=1)
    cv = cross_validate(
        x, y, groups,
        train_cfg=TrainConfig(epochs=1, batch_size=8, seed=0),
        n_splits=4,
    )
    for fold in cv.folds:
        assert set(fold.train_idx).isdisjoint(set(fold.val_idx))


def test_save_load_roundtrip_preserves_predictions(tmp_path):
    x, y, _ = _preprocessed_toy(seed=2)
    model, _, _ = fit_final_model(
        x, y,
        train_cfg=TrainConfig(epochs=3, batch_size=8, seed=0),
    )
    proba_before = model.predict_proba(torch.as_tensor(x[:5], dtype=torch.float32)).numpy()

    path = tmp_path / "model.pt"
    save_model(model, path)
    reloaded = load_model(path)
    proba_after = reloaded.predict_proba(torch.as_tensor(x[:5], dtype=torch.float32)).numpy()

    np.testing.assert_allclose(proba_before, proba_after, atol=1e-6)


def test_modele_lstm_domain_class_end_to_end(tmp_path):
    x, y, groups = _preprocessed_toy(seed=3)
    m = ModeleLSTM()
    cv = m.entrainer(x, y, groups, n_splits=3, epochs=4, batch_size=8, lr=5e-3)
    assert len(cv.folds) == 3
    assert "accuracy_mean" in m.metrics
    assert m.erreur_validation is not None
    assert m.architecture.input_size == x.shape[-1]

    # train a model briefly so we have something to save
    final, _, _ = fit_final_model(x, y, train_cfg=TrainConfig(epochs=2, batch_size=8, seed=0))
    m._model = final
    path = tmp_path / "m.pt"
    m.sauvegarder(path)
    assert path.exists()

    m2 = ModeleLSTM()
    m2.charger(path)
    p1 = m.predire(x[:3])
    p2 = m2.predire(x[:3])
    np.testing.assert_allclose(p1, p2, atol=1e-6)
