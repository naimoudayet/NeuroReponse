"""Train the four models and report them side by side.

    python -m src.models.train_all --root <TDBRAIN root>
    python -m src.models.train_all --sim-only        # no real data needed

The 2x2 design — {simulated, real} x {clinical, clinical+EEG+ECG} — is the
experiment. Reading the four numbers together separates two questions that a
single model cannot: *does the neurophysiological signal add anything* (compare
across a row) and *does the cohort behave as the simulator predicts* (compare
down a column).

Each cohort is loaded **once** and used for both of its variants: reading 132 BDF
recordings twice would double the slowest step for nothing.

Every checkpoint is written with the JSON feature contract that describes how its
inputs were built, so the app can refuse to predict on mismatched data rather
than silently feeding a model the wrong vector.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..data.loader import LoadedDataset
from ..data.modalities import build_features
from ..preprocessing.features import BANDS
from .lstm import LSTMConfig
from .train import TrainConfig, cross_validate, fit_final_model, save_model
from .train_tdbrain import FeatureContract, sidecar_path
from .variants import Dataset, VariantConfig, variants_for


@dataclass
class VariantResult:
    """What one trained variant achieved, for the comparison table."""

    key: str
    label: str
    dataset: str
    modalities: list[str]
    n_patients: int
    n_features: int
    auc_mean: float
    auc_std: float
    accuracy_mean: float
    f1_mean: float
    base_rate: float
    model_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def contract_for(
    cfg: VariantConfig, dataset: LoadedDataset, x: np.ndarray, zscore: bool
) -> FeatureContract:
    features = "+".join(cfg.modalities)
    return FeatureContract(
        source=cfg.dataset.value,
        task="response",
        features=features,
        fs=float(dataset.fs),
        channels=list(dataset.channels or []) if "eeg" in cfg.modalities else [],
        n_bands=len(BANDS),
        per_patient_zscore=bool(zscore),
        input_size=int(x.shape[-1]),
        window=int(dataset.window),
        n_epochs=int(x.shape[1]),
        modalities=list(cfg.modalities),
        ecg_channel="Erbs" if "ecg" in cfg.modalities else None,
        n_rr=int(dataset.ecg.shape[-1]) if ("ecg" in cfg.modalities and dataset.ecg is not None) else 0,
        target=cfg.target,
        protocol=cfg.protocol,
    )


def train_variant(
    cfg: VariantConfig,
    dataset: LoadedDataset,
    n_splits: int = 5,
    train_cfg: TrainConfig | None = None,
    per_patient_zscore: bool = True,
) -> VariantResult:
    """Cross-validate, fit on everything, and persist one variant."""
    x, y, groups, _names = build_features(
        dataset, modalities=cfg.modalities, per_patient_zscore=per_patient_zscore
    )
    train_cfg = train_cfg or TrainConfig()

    cv = cross_validate(
        x, y.astype(np.float32), groups,
        lstm_cfg=LSTMConfig(input_size=x.shape[-1]),
        train_cfg=train_cfg,
        n_splits=n_splits,
    )
    summary = cv.summary()

    model, _, _ = fit_final_model(
        x, y.astype(np.float32),
        lstm_cfg=LSTMConfig(input_size=x.shape[-1]), train_cfg=train_cfg,
    )
    cfg.model.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, cfg.model)

    contract = contract_for(cfg, dataset, x, per_patient_zscore)
    sidecar_path(cfg.model).write_text(
        json.dumps(contract.to_dict(), indent=2), encoding="utf-8"
    )

    return VariantResult(
        key=cfg.key.value,
        label=cfg.label,
        dataset=cfg.dataset.value,
        modalities=list(cfg.modalities),
        n_patients=int(x.shape[0]),
        n_features=int(x.shape[-1]),
        auc_mean=float(summary["auc_mean"]),
        auc_std=float(summary["auc_std"]),
        accuracy_mean=float(summary["accuracy_mean"]),
        f1_mean=float(summary["f1_mean"]),
        base_rate=float(max(y.mean(), 1.0 - y.mean())),
        model_path=str(cfg.model),
    )


def train_dataset(
    dataset_key: Dataset,
    data: LoadedDataset,
    n_splits: int = 5,
    train_cfg: TrainConfig | None = None,
) -> list[VariantResult]:
    """Train every variant belonging to one cohort, from a single load."""
    out = []
    for cfg in variants_for(dataset_key):
        print(f"  training {cfg.key.value} ({'+'.join(cfg.modalities)}) …")
        out.append(train_variant(cfg, data, n_splits=n_splits, train_cfg=train_cfg))
    return out


def comparison_table(results: list[VariantResult]) -> str:
    """The 2x2 as text — the headline artefact of this script."""
    head = (
        f"{'variant':<16}{'cohorte':<11}{'features':>9}{'n':>6}"
        f"{'AUC':>16}{'acc':>8}{'F1':>8}{'base':>8}"
    )
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(
            f"{r.key:<16}{r.dataset:<11}{r.n_features:>9}{r.n_patients:>6}"
            f"{r.auc_mean:>10.3f} ±{r.auc_std:<4.3f}"
            f"{r.accuracy_mean:>8.3f}{r.f1_mean:>8.3f}{r.base_rate:>8.3f}"
        )
    return "\n".join(lines)


def _main() -> None:
    import argparse

    from ..data.simulator_matched import MatchedSimConfig, simulate_matched

    ap = argparse.ArgumentParser(description="Train the four comparison models.")
    ap.add_argument("--root", type=Path, default=None,
                    help="TDBRAIN root; omit (or use --sim-only) to skip the real cohort")
    ap.add_argument("--sim-only", action="store_true")
    ap.add_argument("--effect", type=float, default=0.0,
                    help="simulator effect size (0 reproduces the real null)")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/models/comparison.json"))
    args = ap.parse_args()

    train_cfg = TrainConfig(epochs=args.epochs)
    results: list[VariantResult] = []

    print(f"generating the matched simulated cohort (effect={args.effect}) …")
    sim = simulate_matched(MatchedSimConfig(effect_size=args.effect, seed=args.seed))
    print(f"  {sim.signals_mc.shape[0]} patients · responders "
          f"{int(sim.labels.sum())}/{len(sim.labels)}")
    results += train_dataset(Dataset.SIMULE, sim, args.n_splits, train_cfg)

    if args.root is not None and not args.sim_only:
        from ..data.tdbrain import TDBRAINConfig, load_tdbrain

        print(f"\nloading the real TDBRAIN cohort from {args.root} …")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            real = load_tdbrain(TDBRAINConfig(
                root=args.root, col_id="TDBRAIN_ID", col_protocol="rTMS PROTOCOL",
            ))
        print(f"  {real.signals_mc.shape[0]} patients · responders "
              f"{int(real.labels.sum())}/{len(real.labels)}")
        results += train_dataset(Dataset.TDBRAIN, real, args.n_splits, train_cfg)
    else:
        print("\n(cohorte réelle ignorée — --root non fourni)")

    print("\n" + "=" * 72)
    print("COMPARAISON DES MODÈLES (validation croisée patient-wise)")
    print("=" * 72)
    print(comparison_table(results))
    print(
        "\nNote : 'base' est le taux de la classe majoritaire. Une exactitude "
        "égale à ce taux signifie que le modèle prédit toujours la même classe."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8"
    )
    print(f"\nrésultats -> {args.out}")


if __name__ == "__main__":
    _main()
