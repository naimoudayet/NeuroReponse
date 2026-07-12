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


@dataclass
class LSTMConfig:
    input_size: int
    hidden_sizes: tuple[int, int] = (128, 64)
    dropout: float = 0.3
    bidirectional: bool = False


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
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def predict_tri(self, x: torch.Tensor) -> torch.Tensor:
        """Therapeutic Response Index trajectory in [0, 1], one value per session.

        TRI_t = sigmoid(W h_t + b) — the running estimate of the response probability
        as evidence accumulates session-over-session (NPDT, RES0_AR1 §Bloc 5).
        The final column equals `predict_proba`.
        """
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward_sequence(x))  # (batch, seq)
