"""Does feeding more *treatment sessions* actually help? Measure it.

The clinical loop this app is built around is longitudinal: the patient comes
back, a session is recorded, the model re-predicts on the whole course so far,
the clinician adjusts, repeat. That workflow is only worth the complexity if the
prediction genuinely improves as sessions accumulate — otherwise a single
baseline recording would do, and the loop is theatre.

The 2x2 in :mod:`src.models.train_all` cannot answer this. Both of its cohorts
are **baseline-only**: their sequence axis is epochs of one resting recording, so
"more timesteps" means "more windows of the same two minutes", not "more visits".
The legacy sequential cohort is the only one in this project with a real
treatment trajectory, so it is the only place the question can be asked.

The measurement is deliberately blunt: run the *same* patient-wise
cross-validation on the *same* cohort, truncated to the first ``k`` sessions, for
k = 1..10. Everything except the number of visits is held constant, so the curve
is attributable to the sequence length and nothing else.

**Read the result with the cohort's caveat attached.** ``src/data/simulator.py``
injects a clean alpha-power biomarker, so the absolute AUC is inflated by
construction and says nothing about real rTMS. What the *shape* of the curve
shows is that this pipeline converts extra sessions into accuracy — which is the
architectural claim the loop rests on, and the one thing the negative real-data
result cannot rule on either way.

    python -m src.reporting.sequence_sweep              # k = 1..10
    python -m src.reporting.sequence_sweep --max 5      # shorter
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..data.loader import load
from ..models.lstm import LSTMConfig
from ..models.train import TrainConfig, cross_validate
from ..preprocessing.pipeline import PipelineConfig, preprocess

# Not comparison.json. That file is the 2x2, read by the app as a fixed set of
# four baseline-only variants; appending a fifth row with a different sequence
# axis would put an incomparable number in the same bar chart.
JSON_PATH = Path("data/models/sequence_sweep.json")
SIM_DIR = Path("data/simulated")


@dataclass
class SequencePoint:
    """One cell of the sweep: the cohort cut to ``n_sessions`` visits."""

    n_sessions: int
    auc_mean: float
    auc_std: float
    accuracy_mean: float
    accuracy_std: float
    f1_mean: float
    base_rate: float
    n_patients: int
    n_features: int

    def to_row(self) -> dict:
        return asdict(self)


def load_cohort(data_dir: Path = SIM_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(x, y, groups)`` for the sequential cohort — the same call Training makes.

    Groups are patient indices, so :func:`cross_validate`'s ``GroupKFold`` keeps
    a patient's sessions entirely on one side of every split.
    """
    dataset = load(data_dir)
    x = preprocess(dataset.signals, PipelineConfig(fs=dataset.fs, mode="features")).x
    return x, dataset.labels.astype(np.float32), np.arange(x.shape[0])


def run_point(
    n_sessions: int,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    epochs: int = 30,
    seed: int = 0,
) -> SequencePoint:
    """Cross-validate the cohort truncated to its first ``n_sessions`` visits.

    Truncation is from the front because that is what the clinic sees: after two
    visits you have visits 1 and 2, never the last two. Saves nothing — the
    checkpoints belong to ``train_all``.
    """
    if not 1 <= n_sessions <= x.shape[1]:
        raise ValueError(
            f"n_sessions doit être entre 1 et {x.shape[1]}, reçu {n_sessions}"
        )

    cv = cross_validate(
        x[:, :n_sessions, :], y, groups,
        lstm_cfg=LSTMConfig(input_size=x.shape[-1]),
        train_cfg=TrainConfig(epochs=epochs, seed=seed),
        n_splits=n_splits,
    )
    s = cv.summary()
    return SequencePoint(
        n_sessions=int(n_sessions),
        auc_mean=float(s["auc_mean"]),
        auc_std=float(s["auc_std"]),
        accuracy_mean=float(s["accuracy_mean"]),
        accuracy_std=float(s["accuracy_std"]),
        f1_mean=float(s["f1_mean"]),
        base_rate=float(max(y.mean(), 1.0 - y.mean())),
        n_patients=int(x.shape[0]),
        n_features=int(x.shape[-1]),
    )


def run_sweep(
    max_sessions: int | None = None,
    data_dir: Path = SIM_DIR,
    n_splits: int = 5,
    epochs: int = 30,
    seed: int = 0,
) -> list[SequencePoint]:
    x, y, groups = load_cohort(data_dir)
    top = x.shape[1] if max_sessions is None else min(max_sessions, x.shape[1])
    points = []
    for k in range(1, top + 1):
        print(f"  k={k:2d} séance(s) …", flush=True)
        point = run_point(k, x, y, groups, n_splits=n_splits, epochs=epochs, seed=seed)
        print(f"    AUC {point.auc_mean:.3f} ± {point.auc_std:.3f} "
              f"· exactitude {point.accuracy_mean:.3f}", flush=True)
        points.append(point)
    return points


def write_json(points: list[SequencePoint], path: Path = JSON_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([p.to_row() for p in points], indent=2), encoding="utf-8"
    )
    return path


def read_json(path: Path = JSON_PATH) -> list[dict]:
    """The sweep as the app reads it; empty when it has never been run."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Sweep how many treatment sessions the model is given."
    )
    ap.add_argument("--max", type=int, default=None, dest="max_sessions")
    ap.add_argument("--data", type=Path, default=SIM_DIR)
    ap.add_argument("--out", type=Path, default=JSON_PATH)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    points = run_sweep(
        max_sessions=args.max_sessions, data_dir=args.data,
        n_splits=args.folds, epochs=args.epochs,
    )
    print(f"Écrit : {write_json(points, args.out)}")


if __name__ == "__main__":
    _main()
