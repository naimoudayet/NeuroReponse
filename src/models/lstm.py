"""LSTM architecture for rTMS response prediction.

Matches the design doc: two stacked LSTM layers (128 → 64), dropout between
recurrent layers, and a single sigmoid output for binary classification
(responder vs non-responder).

Input: (batch, n_sessions, n_features)
  - feature mode: n_features = 8 (mean/std/rms + 5 band powers)
  - raw mode:     n_features = window size (e.g. 128)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# What the single output unit means. "classification" -> a logit, squashed to a
# responder probability. "regression" -> a BDI-II point change, read as-is.
CLASSIFICATION = "classification"
REGRESSION = "regression"


@dataclass
class LSTMConfig:
    input_size: int
    hidden_sizes: tuple[int, int] = (128, 64)
    dropout: float = 0.3
    bidirectional: bool = False
    # Defaulted so every checkpoint written before the regression track existed
    # still loads: `load_model` rebuilds the config from the saved dict.
    task: str = CLASSIFICATION

    def __post_init__(self) -> None:
        if self.task not in (CLASSIFICATION, REGRESSION):
            raise ValueError(
                f"task doit être {CLASSIFICATION!r} ou {REGRESSION!r}, reçu {self.task!r}"
            )

    @property
    def is_regression(self) -> bool:
        return self.task == REGRESSION


class ResponseLSTM(nn.Module):
    def __init__(self, cfg: LSTMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h1, h2 = cfg.hidden_sizes
        directions = 2 if cfg.bidirectional else 1

        self.lstm1 = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=h1,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.lstm2 = nn.LSTM(
            input_size=h1 * directions,
            hidden_size=h2,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(h2 * directions, 1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (batch, seq, features), got shape {tuple(x.shape)}")
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        return out  # (batch, seq, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = self._encode(x)[:, -1, :]
        return self.head(last).squeeze(-1)  # logits, shape (batch,)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Per-session logits: the head applied at every timestep, shape (batch, seq)."""
        return self.head(self._encode(x)).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Responder probability. Classification models only."""
        self._require(CLASSIFICATION, "predict_proba")
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def predict_value(self, x: torch.Tensor) -> torch.Tensor:
        """Predicted BDI-II change, shape ``(batch,)``. Regression models only."""
        self._require(REGRESSION, "predict_value")
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def predict_value_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Predicted change after each timestep, shape ``(batch, seq)``.

        The regression counterpart of :meth:`predict_tri`. On a baseline-only
        cohort the axis is epochs of one recording, so this reads as the model
        revising its estimate as it sees more of the same recording — not as a
        clinical trajectory. Same caveat as the TRI, different units.
        """
        self._require(REGRESSION, "predict_value_sequence")
        self.eval()
        with torch.no_grad():
            return self.forward_sequence(x)

    def _require(self, task: str, method: str) -> None:
        """Refuse the wrong head rather than returning a plausible wrong number.

        Calling `predict_proba` on a regression model would push BDI-II *points*
        through a sigmoid: a 12-point improvement becomes "99.999% probability".
        That renders as a perfectly normal-looking curve on every page in the
        app, so it has to raise here, not be caught downstream.
        """
        if self.cfg.task != task:
            raise ValueError(
                f"{method}() exige un modèle « {task} », "
                f"mais ce modèle est « {self.cfg.task} »"
            )

    def predict_tri(self, x: torch.Tensor) -> torch.Tensor:
        """Therapeutic Response Index trajectory in [0, 1], one value per session.

        Classification models only — see :meth:`_require`.

        TRI_t = sigmoid(W h_t + b) — the running estimate of the response probability
        as evidence accumulates session-over-session (NPDT, RES0_AR1 §Bloc 5).
        The final column equals `predict_proba`.
        """
        self._require(CLASSIFICATION, "predict_tri")
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward_sequence(x))  # (batch, seq)
