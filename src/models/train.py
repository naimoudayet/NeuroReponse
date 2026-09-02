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

from .lstm import REGRESSION, LSTMConfig, ResponseLSTM
from .metrics import regression_report


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
    # Regression folds fill these instead of accuracy/auc/f1, which stay nan.
    # One FoldResult type for both tasks keeps the loss curves, split indices and
    # out-of-fold machinery shared; splitting it would duplicate all of that to
    # swap three numbers.
    r: float = float("nan")
    mae: float = float("nan")
    rmse: float = float("nan")
    r2: float = float("nan")
    # Held-out probabilities for this fold's validation patients. Kept so ROC,
    # calibration and confusion charts can be drawn from a single CV run: without
    # them every plot would need its own retraining pass, and any drift between
    # those passes would be invisible.
    val_proba: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass
class CVResult:
    folds: list[FoldResult] = field(default_factory=list)

    def out_of_fold(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Concatenated held-out ``(y_true, y_proba)`` over every fold.

        GroupKFold gives each patient exactly one validation appearance, so this
        is a complete out-of-fold prediction for the cohort — the honest basis for
        an ROC or calibration curve, with no patient scored by a model that saw
        them in training.
        """
        if not self.folds or self.folds[0].val_proba.size == 0:
            raise ValueError(
                "no stored validation probabilities — this CVResult predates "
                "val_proba, re-run cross_validate()"
            )
        # Always returned in **original patient order**, never fold order.
        #
        # Two reasons. With repeated CV each patient is held out once per repeat,
        # and pooling the duplicates would claim n x repeats independent
        # observations — inflating every interval and the permutation test with
        # them; averaging a patient's repeats back to one prediction keeps
        # n = n_patients, the number that is actually independent. And callers
        # line these up against per-patient covariates held in the dataset's own
        # order (`bdi_pre`, age): returning fold order silently correlated
        # mismatched vectors, which turned a baseline of r = 0.500 into r = 0.090
        # and would have flattered every model compared against it.
        idx = np.concatenate([f.val_idx for f in self.folds])
        pred = np.concatenate([f.val_proba for f in self.folds])
        y = np.asarray(y)
        sums = np.zeros(len(y), dtype=np.float64)
        counts = np.zeros(len(y), dtype=np.float64)
        np.add.at(sums, idx, pred)
        np.add.at(counts, idx, 1.0)
        seen = counts > 0
        return y[seen], sums[seen] / counts[seen]

    task: str = "classification"
    repeats: int = 1

    @property
    def is_regression(self) -> bool:
        return self.task == REGRESSION

    def summary(self) -> dict[str, float]:
        """Fold-averaged metrics for whichever task this run was.

        Regression returns r/mae/rmse/r2; classification returns accuracy/auc/f1.
        The keys are disjoint on purpose — a caller that blindly reads
        ``auc_mean`` off a regression result gets a KeyError instead of a nan it
        might average into a comparison table.
        """
        if not self.folds:
            return {}
        if self.is_regression:
            return {
                "r_mean":    float(np.nanmean([f.r for f in self.folds])),
                "r_std":     float(np.nanstd([f.r for f in self.folds])),
                "mae_mean":  float(np.nanmean([f.mae for f in self.folds])),
                "rmse_mean": float(np.nanmean([f.rmse for f in self.folds])),
                "r2_mean":   float(np.nanmean([f.r2 for f in self.folds])),
            }
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
    # MSE on BDI-II points for regression; BCE on logits for classification.
    # Read off the model's own config so a checkpoint can never be trained with
    # the loss belonging to the other head.
    loss_fn = nn.MSELoss() if model.cfg.is_regression else nn.BCEWithLogitsLoss()

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


def _evaluate_regression(
    model: ResponseLSTM, x_val, y_val, device: torch.device
) -> tuple[dict[str, float], np.ndarray]:
    """Regression metrics and the raw predicted values.

    The permutation test is skipped per fold (``n_permutations=0``): a fold holds
    a handful of patients, so its own null is meaningless. The p-value is
    computed once over the pooled out-of-fold predictions, where it has n
    patients behind it.
    """
    model.eval()
    with torch.no_grad():
        pred = model(_to_tensor(x_val).to(device)).cpu().numpy()
    report = regression_report(np.asarray(y_val, dtype=np.float64), pred, n_permutations=0)
    return report, pred


def _evaluate(
    model: ResponseLSTM, x_val, y_val, device: torch.device
) -> tuple[float, float, float, np.ndarray]:
    """Metrics **and** the raw probabilities, so callers can plot from one pass."""
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
    return float(acc), float(auc), float(f1), proba


def _inner_split(
    tr_idx: np.ndarray, groups: np.ndarray, seed: int, fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Carve an early-stopping set out of the **training** fold, patient-wise.

    Early stopping picks which epoch's weights to keep, so whatever it watches has
    been used to fit the model. Watching the outer held-out fold — which
    `cross_validate` used to do — therefore selected each fold's checkpoint on the
    very patients it was then scored on.

    That leak is not academic. With this cohort's near-constant regressor it tuned
    the emitted constant toward each held-out fold's own mean, and the pooled
    out-of-fold correlation reached r = 0.61 on **shuffled** labels, where the
    true signal is zero by construction.

    Split by group, not by row: a patient with several rows must land entirely on
    one side, exactly as the outer GroupKFold guarantees.
    """
    uniq = np.unique(groups[tr_idx])
    if uniq.size < 2:                       # too small to hold anything out
        return tr_idx, tr_idx
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(uniq)[: max(1, int(round(uniq.size * fraction)))].tolist())
    mask = np.array([g in held for g in groups[tr_idx]], dtype=bool)
    if mask.all() or not mask.any():        # degenerate; fall back to no hold-out
        return tr_idx, tr_idx
    return tr_idx[~mask], tr_idx[mask]


def fit_standardiser(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and std over ``(n_patients, n_sessions, n_features)``.

    **Cohort-level, not per-patient**, and the distinction is the whole point.
    Per-patient z-scoring (``zscore_epochs``) centres every patient on their own
    epochs, which subtracts any effect that distinguishes one patient from
    another — measured in ``src.reporting.effect_sweep``, where a planted
    between-patient effect climbs 0.582 -> 0.730 in AUC on raw features and stays
    flat at 0.46 on the z-scored ones. Responder status is a between-patient
    label, so it is exactly what gets subtracted.

    This standardiser preserves between-patient variance and only equalises the
    *units*, which is what a network of mixed feature families needs: PLV lives
    on [0, 1], SDNN in the tens of milliseconds, alpha peak near 10 Hz. Without
    it the optimiser spends its budget on whichever block happens to be largest.
    """
    flat = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
    # A constant or all-NaN column divides by 1 instead of by 0: it carries no
    # information either way, and blowing up here would take the whole fold down.
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    return mean, std


def apply_standardiser(
    x: np.ndarray, stats: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Apply :func:`fit_standardiser`'s statistics; non-finite entries become 0."""
    mean, std = stats
    out = (np.asarray(x, dtype=np.float64) - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def cross_validate(
    x: np.ndarray,           # (n_patients, n_sessions, n_features)
    y: np.ndarray,           # (n_patients,) 0/1
    groups: np.ndarray,      # (n_patients,) patient ids
    lstm_cfg: LSTMConfig | None = None,
    train_cfg: TrainConfig | None = None,
    n_splits: int = 5,
    repeats: int = 1,
    standardise: bool = False,
) -> CVResult:
    """Patient-wise cross-validation, for either task.

    ``repeats`` reruns the whole CV with a different group-to-fold assignment,
    as the reference study does (10 repetitions of 10-fold). On a 44-patient
    protocol a single 5-fold split puts ~9 patients in each validation set, so
    the metric depends heavily on which nine — repeating and averaging is what
    makes the number mean something.

    ``repeats=1`` keeps the *unshuffled* splitter, so every result recorded in
    this project before repeated CV existed reproduces exactly. Shuffling is
    switched on only when there is more than one repeat to distinguish, since
    without it every repeat would return the identical split.

    ``standardise`` scales each feature by statistics computed **on the training
    fold only** — never on the whole cohort, which would leak the held-out
    patients' scale into the model that scores them. It defaults to off so every
    previously recorded figure reproduces byte-for-byte; the network feature
    blocks (``sync`` / ``cplx`` / ``h369``) need it, because they mix quantities
    ranging over three orders of magnitude. See :func:`fit_standardiser` for why
    this is not the same operation as per-patient z-scoring.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be (n_patients, n_sessions, n_features); got {x.shape}")
    if repeats < 1:
        raise ValueError(f"repeats doit être >= 1, reçu {repeats}")
    lstm_cfg = lstm_cfg or LSTMConfig(input_size=x.shape[-1])
    train_cfg = train_cfg or TrainConfig()
    _seed_everything(train_cfg.seed)

    result = CVResult(task=lstm_cfg.task, repeats=int(repeats))
    device = torch.device(train_cfg.device)
    fold_id = 0

    for repeat in range(repeats):
        splitter = (
            GroupKFold(n_splits=n_splits)
            if repeats == 1
            else GroupKFold(n_splits=n_splits, shuffle=True,
                            random_state=train_cfg.seed + repeat)
        )
        for tr_idx, va_idx in splitter.split(x, y, groups=groups):
            _seed_everything(train_cfg.seed + fold_id)
            model = ResponseLSTM(lstm_cfg)
            # Early stopping watches an inner split of the training fold. The
            # outer fold stays untouched until scoring, or the metric measures a
            # model that was tuned on its own test set.
            fit_idx, stop_idx = _inner_split(
                tr_idx, np.asarray(groups), train_cfg.seed + fold_id
            )
            # Scale on the training fold only. Fitting the standardiser on `x`
            # entire would let the held-out patients' mean and spread reach the
            # model that is about to be scored on them — a quieter cousin of the
            # early-stopping leak `_inner_split` exists to prevent.
            x_fit, x_stop, x_val = x[fit_idx], x[stop_idx], x[va_idx]
            if standardise:
                stats = fit_standardiser(x_fit)
                x_fit = apply_standardiser(x_fit, stats)
                x_stop = apply_standardiser(x_stop, stats)
                x_val = apply_standardiser(x_val, stats)
            train_losses, val_losses, best_epoch = train_one_fold(
                model,
                x_fit, y[fit_idx].astype(np.float32),
                x_stop, y[stop_idx].astype(np.float32),
                train_cfg,
            )
            if lstm_cfg.is_regression:
                report, pred = _evaluate_regression(model, x_val, y[va_idx], device)
                fold_result = FoldResult(
                    fold=fold_id, train_idx=tr_idx, val_idx=va_idx,
                    train_losses=train_losses, val_losses=val_losses,
                    accuracy=float("nan"), auc=float("nan"), f1=float("nan"),
                    best_epoch=best_epoch,
                    r=report["r"], mae=report["mae"],
                    rmse=report["rmse"], r2=report["r2"],
                    val_proba=np.asarray(pred, dtype=np.float64),
                )
            else:
                acc, auc, f1, proba = _evaluate(model, x_val, y[va_idx], device)
                fold_result = FoldResult(
                    fold=fold_id, train_idx=tr_idx, val_idx=va_idx,
                    train_losses=train_losses, val_losses=val_losses,
                    accuracy=acc, auc=auc, f1=f1, best_epoch=best_epoch,
                    val_proba=np.asarray(proba, dtype=np.float64),
                )
            result.folds.append(fold_result)
            fold_id += 1
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
