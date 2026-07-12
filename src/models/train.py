"""Training + patient-wise cross-validation for the response LSTM.

The split logic is the critical piece of the doc: patients in the train set
must NEVER appear in val/test. We use sklearn's GroupKFold with patient_id
as the group key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from .lstm import LSTMConfig, ResponseLSTM


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 5
    device: str = "cpu"
    seed: int = 0


@dataclass
class FoldResult:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_losses: list[float]
    val_losses: list[float]
    accuracy: float
    auc: float
    f1: float
    best_epoch: int


@dataclass
class CVResult:
    folds: list[FoldResult] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        if not self.folds:
            return {}
        return {
            "accuracy_mean": float(np.mean([f.accuracy for f in self.folds])),
            "accuracy_std":  float(np.std([f.accuracy for f in self.folds])),
            "auc_mean":      float(np.mean([f.auc for f in self.folds])),
            "auc_std":       float(np.std([f.auc for f in self.folds])),
            "f1_mean":       float(np.mean([f.f1 for f in self.folds])),
            "f1_std":        float(np.std([f.f1 for f in self.folds])),
        }


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_tensor(x: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype)


def _build_loaders(x_tr, y_tr, x_va, y_va, batch_size: int) -> tuple[DataLoader, DataLoader]:
    tr = TensorDataset(_to_tensor(x_tr), _to_tensor(y_tr))
    va = TensorDataset(_to_tensor(x_va), _to_tensor(y_va))
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True),
        DataLoader(va, batch_size=batch_size, shuffle=False),
    )


def train_one_fold(
    model: ResponseLSTM,
    x_train, y_train, x_val, y_val,
    cfg: TrainConfig,
) -> tuple[list[float], list[float], int]:
    device = torch.device(cfg.device)
    model.to(device)

    train_loader, val_loader = _build_loaders(x_train, y_train, x_val, y_val, cfg.batch_size)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    patience = 0

    for epoch in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
            tr_n += xb.size(0)
        train_losses.append(tr_loss / max(tr_n, 1))

        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                va_loss += loss.item() * xb.size(0)
                va_n += xb.size(0)
        avg_val = va_loss / max(va_n, 1)
        val_losses.append(avg_val)

        if avg_val < best_val - 1e-4:
            best_val = avg_val
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                break

    model.load_state_dict(best_state)
    return train_losses, val_losses, best_epoch


def _evaluate(model: ResponseLSTM, x_val, y_val, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(_to_tensor(x_val).to(device))
        proba = torch.sigmoid(logits).cpu().numpy()
    pred = (proba >= 0.5).astype(int)
    y_val = np.asarray(y_val).astype(int)
    acc = accuracy_score(y_val, pred)
    f1 = f1_score(y_val, pred, zero_division=0)
    try:
        auc = roc_auc_score(y_val, proba)
    except ValueError:
        auc = float("nan")  # all-one-class fold
    return float(acc), float(auc), float(f1)


def cross_validate(
    x: np.ndarray,           # (n_patients, n_sessions, n_features)
    y: np.ndarray,           # (n_patients,) 0/1
    groups: np.ndarray,      # (n_patients,) patient ids
    lstm_cfg: LSTMConfig | None = None,
    train_cfg: TrainConfig | None = None,
    n_splits: int = 5,
) -> CVResult:
    if x.ndim != 3:
        raise ValueError(f"x must be (n_patients, n_sessions, n_features); got {x.shape}")
    lstm_cfg = lstm_cfg or LSTMConfig(input_size=x.shape[-1])
    train_cfg = train_cfg or TrainConfig()
    _seed_everything(train_cfg.seed)

    splitter = GroupKFold(n_splits=n_splits)
    result = CVResult()

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y, groups=groups)):
        _seed_everything(train_cfg.seed + fold)
        model = ResponseLSTM(lstm_cfg)
        train_losses, val_losses, best_epoch = train_one_fold(
            model,
            x[tr_idx], y[tr_idx].astype(np.float32),
            x[va_idx], y[va_idx].astype(np.float32),
            train_cfg,
        )
        acc, auc, f1 = _evaluate(model, x[va_idx], y[va_idx], torch.device(train_cfg.device))
        result.folds.append(
            FoldResult(
                fold=fold,
                train_idx=tr_idx,
                val_idx=va_idx,
                train_losses=train_losses,
                val_losses=val_losses,
                accuracy=acc,
                auc=auc,
                f1=f1,
                best_epoch=best_epoch,
            )
        )
    return result


def fit_final_model(
    x: np.ndarray,
    y: np.ndarray,
    lstm_cfg: LSTMConfig | None = None,
    train_cfg: TrainConfig | None = None,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[ResponseLSTM, list[float], list[float]]:
    """Train on the full dataset (with a small internal hold-out for early stopping)."""
    lstm_cfg = lstm_cfg or LSTMConfig(input_size=x.shape[-1])
    train_cfg = train_cfg or TrainConfig()
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    model = ResponseLSTM(lstm_cfg)
    tr_losses, va_losses, _ = train_one_fold(
        model,
        x[tr_idx], y[tr_idx].astype(np.float32),
        x[val_idx], y[val_idx].astype(np.float32),
        train_cfg,
    )
    return model, tr_losses, va_losses


def save_model(model: ResponseLSTM, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.cfg.__dict__}, path)


def load_model(path: Path) -> ResponseLSTM:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = LSTMConfig(**payload["config"])
    model = ResponseLSTM(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
