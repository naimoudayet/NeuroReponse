"""Regenerate the static figures embedded in README.md / PFE report.

Usage:
    python -m src.reporting.figures               # writes docs/figures/*.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import welch
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import GroupKFold

from ..data.loader import load
from ..models.lstm import LSTMConfig, ResponseLSTM
from ..models.train import TrainConfig, _to_tensor, cross_validate, train_one_fold
from ..preprocessing.pipeline import PipelineConfig, preprocess


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_alpha_trajectory(signals: np.ndarray, labels: np.ndarray, fs: float, out_dir: Path) -> Path:
    freqs, psd = welch(signals, fs=fs, nperseg=signals.shape[-1], axis=-1)
    alpha = psd[..., (freqs >= 8) & (freqs <= 13)].mean(axis=-1)
    sessions = np.arange(signals.shape[1])

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color, name, marker in [(1, "#3b82f6", "Répondeurs", "o"),
                                       (0, "#64748b", "Non-répondeurs", "s")]:
        mean = alpha[labels == label].mean(0)
        std = alpha[labels == label].std(0)
        ax.plot(sessions, mean, marker=marker, color=color, label=name)
        ax.fill_between(sessions, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel("Indice de séance")
    ax.set_ylabel("Puissance alpha (8–13 Hz)")
    ax.set_title("Évolution de la puissance alpha par groupe")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "alpha_trajectory.png")


def figure_cv_metrics(cv, out_dir: Path) -> Path:
    metrics = np.array([[f.accuracy, f.auc, f.f1] for f in cv.folds])
    x = np.arange(len(cv.folds))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.25, metrics[:, 0], 0.25, label="Accuracy", color="#3b82f6")
    ax.bar(x,        metrics[:, 1], 0.25, label="AUC",      color="#10b981")
    ax.bar(x + 0.25, metrics[:, 2], 0.25, label="F1",       color="#f59e0b")
    ax.axhline(0.5, color="#94a3b8", ls="--", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f.fold}" for f in cv.folds])
    ax.set_ylim(0, 1.05)
    ax.set_title("Métriques par fold — validation patient-wise")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir, "cv_metrics.png")


def figure_diagnostics(y: np.ndarray, oof_proba: np.ndarray, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    fpr, tpr, _ = roc_curve(y.astype(int), oof_proba)
    axes[0].plot(fpr, tpr, color="#3b82f6", lw=2)
    axes[0].plot([0, 1], [0, 1], "--", color="#94a3b8")
    axes[0].set_xlabel("Faux positifs")
    axes[0].set_ylabel("Vrais positifs")
    axes[0].set_title("Courbe ROC (out-of-fold)")
    axes[0].grid(alpha=0.3)

    cm = confusion_matrix(y.astype(int), (oof_proba >= 0.5).astype(int))
    axes[1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Non-rép.", "Rép."])
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Non-rép.", "Rép."])
    axes[1].set_xlabel("Prédit")
    axes[1].set_ylabel("Vrai")
    axes[1].set_title("Matrice de confusion")

    axes[2].hist(oof_proba[y == 0], bins=20, alpha=0.6, color="#64748b", label="Non-rép.")
    axes[2].hist(oof_proba[y == 1], bins=20, alpha=0.6, color="#3b82f6", label="Rép.")
    axes[2].axvline(0.5, color="#ef4444", ls="--", label="seuil 0.5")
    axes[2].set_xlabel("Probabilité prédite")
    axes[2].set_ylabel("Patients")
    axes[2].set_title("Distribution OOF")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    return _save(fig, out_dir, "diagnostics.png")


def _oof_predictions(X, y, groups, lstm_cfg, train_cfg, n_splits=5):
    oof = np.zeros(len(y))
    for fold_idx, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups=groups)):
        model = ResponseLSTM(lstm_cfg)
        train_one_fold(model, X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
                       TrainConfig(**{**train_cfg.__dict__, "seed": fold_idx}))
        with torch.no_grad():
            oof[va_idx] = torch.sigmoid(model(_to_tensor(X[va_idx]))).numpy()
    return oof


def main(out_dir: Path = Path("docs/figures")) -> list[Path]:
    ds = load()
    pre = preprocess(ds.signals, PipelineConfig(fs=ds.fs, mode="features"))
    X, y = pre.x, ds.labels.astype(np.float32)
    groups = np.arange(X.shape[0])

    lstm_cfg = LSTMConfig(input_size=X.shape[-1], hidden_sizes=(128, 64), dropout=0.3)
    train_cfg = TrainConfig(epochs=40, batch_size=8, lr=3e-3, early_stopping_patience=6, seed=0)

    cv = cross_validate(X, y, groups=groups, lstm_cfg=lstm_cfg, train_cfg=train_cfg, n_splits=5)
    oof = _oof_predictions(X, y, groups, lstm_cfg, train_cfg, n_splits=5)

    paths = [
        figure_alpha_trajectory(ds.signals, ds.labels, ds.fs, out_dir),
        figure_cv_metrics(cv, out_dir),
        figure_diagnostics(y, oof, out_dir),
    ]
    for p in paths:
        print(f"  wrote {p}")
    return paths


if __name__ == "__main__":
    main()
