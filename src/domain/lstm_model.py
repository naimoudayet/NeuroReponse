from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ArchitectureSpec:
    lstm_units: tuple[int, int] = (128, 64)
    dropout: float = 0.3
    input_size: int = 8  # default = number of session-level features
    bidirectional: bool = False


@dataclass
class ModeleLSTM:
    architecture: ArchitectureSpec = field(default_factory=ArchitectureSpec)
    poids_path: Path | None = None
    erreur_validation: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    _model: Any = field(default=None, repr=False)

    def _ensure_model(self) -> Any:
        if self._model is None:
            from ..models.lstm import LSTMConfig, ResponseLSTM

            self._model = ResponseLSTM(
                LSTMConfig(
                    input_size=self.architecture.input_size,
                    hidden_sizes=self.architecture.lstm_units,
                    dropout=self.architecture.dropout,
                    bidirectional=self.architecture.bidirectional,
                )
            )
        return self._model

    def entrainer(
        self,
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        n_splits: int = 5,
        epochs: int = 30,
        batch_size: int = 16,
        lr: float = 1e-3,
    ):
        from ..models.lstm import LSTMConfig
        from ..models.train import TrainConfig, cross_validate

        cv = cross_validate(
            x,
            y,
            groups,
            lstm_cfg=LSTMConfig(
                input_size=x.shape[-1],
                hidden_sizes=self.architecture.lstm_units,
                dropout=self.architecture.dropout,
                bidirectional=self.architecture.bidirectional,
            ),
            train_cfg=TrainConfig(epochs=epochs, batch_size=batch_size, lr=lr),
            n_splits=n_splits,
        )
        self.metrics = cv.summary()
        self.erreur_validation = float(np.mean([f.val_losses[f.best_epoch] for f in cv.folds]))
        self.architecture.input_size = x.shape[-1]
        return cv

    def predire(self, x: np.ndarray) -> np.ndarray:
        import torch

        model = self._ensure_model()
        with torch.no_grad():
            logits = model(torch.as_tensor(x, dtype=torch.float32))
            return torch.sigmoid(logits).numpy()

    def evaluer(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        proba = self.predire(x)
        pred = (proba >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "auc": float(roc_auc_score(y, proba)) if len(set(y)) > 1 else float("nan"),
            "f1": float(f1_score(y, pred, zero_division=0)),
        }

    def sauvegarder(self, path: Path) -> None:
        from ..models.train import save_model

        save_model(self._ensure_model(), path)
        self.poids_path = path

    def charger(self, path: Path) -> None:
        from ..models.train import load_model

        self._model = load_model(path)
        self.architecture.input_size = self._model.cfg.input_size
        self.architecture.lstm_units = self._model.cfg.hidden_sizes
        self.architecture.dropout = self._model.cfg.dropout
        self.architecture.bidirectional = self._model.cfg.bidirectional
        self.poids_path = path
