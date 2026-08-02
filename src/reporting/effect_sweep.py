"""The positive control: sweep the simulator's effect size and watch AUC follow.

The central result of this project is negative — rTMS response is at chance on
TDBRAIN. A negative result is only worth anything if the pipeline that produced
it can be shown to detect an effect *when one exists*. That is what this sweep
is for.

``simulate_matched`` reproduces the real cohort's shape and statistics exactly,
with one knob: ``effect_size``, the only route by which label information enters
the neurophysiological blocks. Running the **same** feature construction and the
**same** cross-validation across a range of effect sizes gives a calibration
curve. At 0 it must land on chance (reproducing the real null); as the effect
grows it must climb. A pipeline that stayed flat would be broken; one that scored
well at 0 would be leaking.

Nothing here writes a checkpoint. The four trained models and
``comparison.json`` are artefacts of :mod:`src.models.train_all`, and a sweep
that overwrote them with effect-injected weights would silently corrupt the app.

    python -m src.reporting.effect_sweep                  # default sweep
    python -m src.reporting.effect_sweep --effects 0 0.3  # fewer points
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..data.modalities import build_features
from ..data.simulator_matched import MatchedSimConfig, simulate_matched
from ..models.lstm import LSTMConfig
from ..models.train import TrainConfig, cross_validate
from . import model_charts as mc

# Three arms of (feature set, per-patient z-scoring), because the normalisation
# is not a detail here — it decides whether the injected effect is visible at all.
#
# The simulator raises alpha for responders by a **per-patient constant**: every
# epoch of that patient is shifted by the same amount. Per-patient z-scoring
# centres each patient on their own epochs, so it subtracts exactly that constant
# and deletes the effect. The raw arm is therefore the true positive control; the
# z-scored arm measures what the project's default preprocessing throws away.
ARMS: tuple[tuple[tuple[str, ...], bool], ...] = (
    (("eeg",), False),
    (("eeg",), True),
    (("rtms", "eeg", "ecg"), True),
)
DEFAULT_EFFECTS: tuple[float, ...] = (0.0, 0.15, 0.30, 0.50)

CSV_PATH = Path("docs/strategy/effect_sweep.csv")
FIG_PATH = Path("docs/figures")


@dataclass
class SweepPoint:
    features: str
    zscore: bool
    effect_size: float
    auc_mean: float
    auc_std: float
    accuracy_mean: float
    f1_mean: float
    base_rate: float
    n_patients: int
    n_features: int

    def to_row(self) -> dict:
        return self.__dict__.copy()


def run_point(
    effect: float,
    modalities: tuple[str, ...] = ("eeg",),
    zscore: bool = True,
    n_splits: int = 5,
    epochs: int = 30,
    seed: int = 42,
    sim: MatchedSimConfig | None = None,
) -> SweepPoint:
    """Cross-validate one (effect, feature set, normalisation) cell, saving nothing.

    ``sim`` overrides the cohort shape — the defaults reproduce TDBRAIN, which is
    what the report needs, but tests want a handful of short recordings.
    """
    base = sim or MatchedSimConfig()
    dataset = simulate_matched(replace(base, effect_size=effect, seed=seed))
    x, y, groups, _names = build_features(
        dataset, modalities=modalities, per_patient_zscore=zscore
    )
    y = y.astype(np.float32)

    cv = cross_validate(
        x, y, groups,
        lstm_cfg=LSTMConfig(input_size=x.shape[-1]),
        train_cfg=TrainConfig(epochs=epochs),
        n_splits=n_splits,
    )
    s = cv.summary()
    return SweepPoint(
        features="+".join(modalities),
        zscore=bool(zscore),
        effect_size=float(effect),
        auc_mean=float(s["auc_mean"]),
        auc_std=float(s["auc_std"]),
        accuracy_mean=float(s["accuracy_mean"]),
        f1_mean=float(s["f1_mean"]),
        base_rate=float(max(y.mean(), 1.0 - y.mean())),
        n_patients=int(x.shape[0]),
        n_features=int(x.shape[-1]),
    )


def run_sweep(
    effects: tuple[float, ...] = DEFAULT_EFFECTS,
    arms: tuple[tuple[tuple[str, ...], bool], ...] = ARMS,
    n_splits: int = 5,
    epochs: int = 30,
    seed: int = 42,
) -> list[SweepPoint]:
    points = []
    for modalities, zscore in arms:
        name = f"{'+'.join(modalities)} {'z' if zscore else 'brut'}"
        for effect in effects:
            print(f"  [{name}] effect={effect:.2f} …", flush=True)
            point = run_point(effect, modalities, zscore, n_splits=n_splits,
                              epochs=epochs, seed=seed)
            print(f"    AUC {point.auc_mean:.3f} ± {point.auc_std:.3f}", flush=True)
            points.append(point)
    return points


def write_csv(points: list[SweepPoint], path: Path = CSV_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(points[0].to_row()))
        writer.writeheader()
        writer.writerows(p.to_row() for p in points)
    return path


def read_csv(path: Path = CSV_PATH) -> list[SweepPoint]:
    """Load a previous sweep, so the figure can be redrawn without retraining."""
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [
        SweepPoint(
            features=r["features"],
            zscore=r["zscore"] == "True",
            effect_size=float(r["effect_size"]),
            auc_mean=float(r["auc_mean"]),
            auc_std=float(r["auc_std"]),
            accuracy_mean=float(r["accuracy_mean"]),
            f1_mean=float(r["f1_mean"]),
            base_rate=float(r["base_rate"]),
            n_patients=int(r["n_patients"]),
            n_features=int(r["n_features"]),
        )
        for r in rows
    ]


LIBELLE = {
    ("eeg", False): "EEG brut (130)",
    ("eeg", True): "EEG z-scoré (130)",
    ("rtms+eeg+ecg", True): "Multimodal z-scoré (139)",
    ("rtms+eeg+ecg", False): "Multimodal brut (139)",
}


def figure_effect_curve(points: list[SweepPoint], out_dir: Path = FIG_PATH) -> Path:
    """AUC against injected effect size, one curve per arm.

    Direct labels rather than a legend: the curves converge at the left edge, so a
    legend would force the reader to match colours back and forth.
    """
    series: dict[tuple[str, bool], list[SweepPoint]] = {}
    for point in points:
        series.setdefault((point.features, point.zscore), []).append(point)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fig.patch.set_facecolor(mc.SURFACE)
    ax.set_facecolor(mc.SURFACE)

    all_effects = sorted({p.effect_size for p in points})
    for i, (name, pts) in enumerate(series.items()):
        pts = sorted(pts, key=lambda p: p.effect_size)
        colour = mc.SERIES[i % len(mc.SERIES)]
        # Error bars kept — the fold spread is the caveat — but drawn faintly and
        # slightly offset, so three overlapping sets do not bury the curves.
        offset = (i - (len(series) - 1) / 2) * 0.008
        ax.errorbar(
            [p.effect_size + offset for p in pts], [p.auc_mean for p in pts],
            yerr=[p.auc_std for p in pts], fmt="none",
            ecolor=colour, elinewidth=1, capsize=3, alpha=0.28, zorder=2,
        )
        ax.plot(
            [p.effect_size for p in pts], [p.auc_mean for p in pts],
            marker="o", markersize=6, color=colour, linewidth=2, zorder=3,
        )
        ax.annotate(
            LIBELLE.get(name, f"{name[0]}{' z' if name[1] else ''}"),
            xy=(pts[-1].effect_size, pts[-1].auc_mean), xytext=(8, -2),
            textcoords="offset points", color=colour, fontsize=9,
            fontweight="bold", va="center",
        )

    ax.axhline(mc.CHANCE, color=mc.MUTED, linestyle="--", linewidth=1.2, zorder=1)
    # Right-hand side and above the line: the left edge is where the flat
    # z-scored curve sits, a few hundredths below chance.
    ax.text(
        0.62, mc.CHANCE + 0.012, "hasard (0.5)",
        transform=ax.get_yaxis_transform(which="grid"),
        ha="left", va="bottom", color=mc.INK_2, fontsize=9,
    )

    mc.style_axes(
        ax,
        title="Contrôle positif : le pipeline détecte un effet quand il existe",
        xlabel="Effet injecté dans l'EEG/ECG (écarts-types entre patients)",
        ylabel="AUC (validation croisée patient-wise)",
    )
    ax.set_ylim(0.35, 1.02)
    ax.set_xticks(all_effects)
    ax.set_xlim(-0.03, max(all_effects) * 1.22)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "effect_size_curve.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--effects", type=float, nargs="+", default=list(DEFAULT_EFFECTS))
    ap.add_argument("--arms", nargs="+",
                    default=[f"{'+'.join(m)}:{'z' if z else 'raw'}" for m, z in ARMS],
                    help="e.g. eeg:raw eeg:z rtms+eeg+ecg:z")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--from-csv", action="store_true",
                    help="redraw the figure from the stored sweep, training nothing")
    args = ap.parse_args()

    if args.from_csv:
        points = read_csv()
        print(f"figure -> {figure_effect_curve(points)}")
        return

    arms = []
    for spec in args.arms:
        features, _, mode = spec.partition(":")
        arms.append((tuple(features.split("+")), mode != "raw"))

    print(f"sweeping effect sizes {args.effects} on {args.arms} …")
    points = run_sweep(
        tuple(args.effects), tuple(arms),
        n_splits=args.n_splits, epochs=args.epochs, seed=args.seed,
    )

    print(f"\n{'features':<16}{'norm':<6}{'effect':>7}   {'AUC':<16} {'acc':>7} {'base':>7}")
    for p in points:
        print(f"{p.features:<16}{'z' if p.zscore else 'brut':<6}{p.effect_size:>7.2f}   "
              f"{p.auc_mean:.3f} ± {p.auc_std:.3f}    "
              f"{p.accuracy_mean:.3f}   {p.base_rate:.3f}")

    print(f"\ncsv    -> {write_csv(points)}")
    print(f"figure -> {figure_effect_curve(points)}")


if __name__ == "__main__":
    _main()
