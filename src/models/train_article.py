"""Train the article-aligned arm: continuous BDI-II change, one model per protocol.

This mirrors Arteaga et al. (PMC12981298), which predicts the *change* in BDI-II
rather than a binary responder label, fits **separate models for the two rTMS
protocols**, and scores with Pearson's r against a permutation null.

    python -m src.models.train_article --sim-only          # negative control only
    python -m src.models.train_article --seed-db recherche_tdbrain.sqlite3

Kept separate from :mod:`src.models.train_all` on purpose. That script produces
the project's published 2x2 and its ``comparison.json``; r and AUC are not
comparable quantities, so mixing them into one table or one artefact would invite
exactly the misreading the app is careful to avoid everywhere else.

**Every multimodal row is reported next to its clinical-only twin, and that is
not optional.** ``delta_bdi`` is mathematically coupled to baseline severity --
you cannot recover 40 points from a BDI of 20. On this cohort ``bdi_pre`` alone
reaches r = 0.500 on protocol 1 — the same magnitude as the article's headline
r = 0.401 from EEG, and with a bootstrap CI of [0.258, 0.700] indistinguishable
from it. A neurophysiological model that cannot be told apart from its clinical
baseline has demonstrated nothing, so the baseline travels in the same result row.

The ``r_partial`` column attacks the same coupling from the other direction: it
correlates prediction with truth *after projecting baseline severity out of
both*, so it reports only what the model found that the intake BDI-II did not
already contain.

An earlier attempt at that column divided both truth and prediction by
``bdi_pre`` and correlated the ratios. That is wrong, and expensively so: a
shared divisor manufactures correlation, and on this cohort's shape a model
emitting a **constant** prediction scored 0.539 by it. Partial correlation
returns ~0 for that same model. Do not reintroduce the ratio.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..data.loader import LoadedDataset
from ..data.modalities import build_features, protocol_mask
from ..preprocessing.features import BANDS
from .lstm import REGRESSION, LSTMConfig
from .metrics import (
    N_PERMUTATIONS,
    partial_correlation,
    pearson_r,
    regression_report,
)
from .train import TrainConfig, cross_validate, fit_final_model, save_model
from .train_tdbrain import FeatureContract, sidecar_path
from .variants import Dataset, Task, VariantConfig, article_variants

OUT_PATH = Path("data/models/article_comparison.json")


@dataclass
class ArticleResult:
    """One article-aligned variant, with the baseline it has to beat."""

    key: str
    label: str
    dataset: str
    protocol: int | None
    modalities: list[str]
    target: str
    n_patients: int
    n_features: int
    # Fold-averaged, then pooled out-of-fold (the article reports the latter kind).
    r_mean: float
    r_std: float
    r_oof: float
    p_perm: float
    mae: float
    rmse: float
    r2: float
    # Correlation with baseline severity partialled out -- the signal that is
    # not simply 'sicker patients have more room to improve'.
    r_partial: float
    # Trivial baselines: single clinical covariates, no model fitted.
    baseline_r_bdi_pre: float
    baseline_r_age: float
    beats_baseline: bool
    model_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def _covariate(dataset: LoadedDataset, name: str, mask: np.ndarray) -> np.ndarray | None:
    meta = dataset.metadata
    if name not in meta:
        return None
    return meta[name].to_numpy(dtype=np.float64)[mask]


def _contract(
    cfg: VariantConfig, dataset: LoadedDataset, x: np.ndarray, zscore: bool
) -> FeatureContract:
    has_ecg = "ecg" in cfg.modalities and dataset.ecg is not None
    return FeatureContract(
        source=cfg.dataset.value,
        task="response",
        features="+".join(cfg.modalities),
        fs=float(dataset.fs),
        channels=list(dataset.channels or []) if "eeg" in cfg.modalities else [],
        n_bands=len(BANDS),
        per_patient_zscore=bool(zscore),
        input_size=int(x.shape[-1]),
        window=int(dataset.window),
        n_epochs=int(x.shape[1]),
        modalities=list(cfg.modalities),
        ecg_channel="Erbs" if "ecg" in cfg.modalities else None,
        n_rr=int(dataset.ecg.shape[-1]) if has_ecg else 0,
        target=cfg.target,
        protocol=cfg.protocol,
    )


def train_regression_variant(
    cfg: VariantConfig,
    dataset: LoadedDataset,
    n_splits: int = 5,
    repeats: int = 1,
    train_cfg: TrainConfig | None = None,
    per_patient_zscore: bool = True,
    n_permutations: int = N_PERMUTATIONS,
) -> ArticleResult:
    """Cross-validate, fit on everything, persist, and score against the baseline."""
    if cfg.task is not Task.REGRESSION:
        raise ValueError(f"{cfg.key.value} is not a regression variant")

    x, y, groups, _names = build_features(
        dataset, modalities=cfg.modalities, per_patient_zscore=per_patient_zscore,
        target=cfg.target, protocol=cfg.protocol,
    )
    y = y.astype(np.float32)
    train_cfg = train_cfg or TrainConfig()
    lstm_cfg = LSTMConfig(input_size=x.shape[-1], task=REGRESSION)

    cv = cross_validate(
        x, y, groups, lstm_cfg=lstm_cfg, train_cfg=train_cfg,
        n_splits=n_splits, repeats=repeats,
    )
    summary = cv.summary()

    # Pooled out-of-fold: every patient scored exactly once by a model that never
    # saw them. This is the honest n for a permutation test; per-fold r on ~9
    # patients is far too noisy to permute meaningfully.
    y_true, y_pred = cv.out_of_fold(y)
    pooled = regression_report(y_true, y_pred, n_permutations=n_permutations)

    mask = protocol_mask(dataset, cfg.protocol)
    bdi_pre = _covariate(dataset, "bdi_pre", mask)
    age = _covariate(dataset, "age", mask)

    # Partial correlation controlling for baseline severity: what the model found
    # that `bdi_pre` did not already contain. NOT a ratio of the two divided by
    # bdi_pre -- that shared divisor manufactures correlation, and scored 0.539
    # for a constant predictor when it was tried. See `partial_correlation`.
    r_partial = float("nan")
    if bdi_pre is not None:
        r_partial = partial_correlation(y_true, y_pred, bdi_pre)

    base_bdi = pearson_r(y_true, bdi_pre) if bdi_pre is not None else float("nan")
    base_age = pearson_r(y_true, age) if age is not None else float("nan")

    model, _, _ = fit_final_model(x, y, lstm_cfg=lstm_cfg, train_cfg=train_cfg)
    cfg.model.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, cfg.model)
    sidecar_path(cfg.model).write_text(
        json.dumps(_contract(cfg, dataset, x, per_patient_zscore).to_dict(), indent=2),
        encoding="utf-8",
    )

    return ArticleResult(
        key=cfg.key.value, label=cfg.label, dataset=cfg.dataset.value,
        protocol=cfg.protocol, modalities=list(cfg.modalities), target=cfg.target,
        n_patients=int(x.shape[0]), n_features=int(x.shape[-1]),
        r_mean=float(summary["r_mean"]), r_std=float(summary["r_std"]),
        r_oof=float(pooled["r"]), p_perm=float(pooled["p_perm"]),
        mae=float(pooled["mae"]), rmse=float(pooled["rmse"]), r2=float(pooled["r2"]),
        r_partial=float(r_partial),
        baseline_r_bdi_pre=float(base_bdi), baseline_r_age=float(base_age),
        # nan-safe: an unmeasurable r never counts as beating anything.
        beats_baseline=bool(
            np.isfinite(pooled["r"]) and np.isfinite(base_bdi)
            and pooled["r"] > base_bdi
        ),
        model_path=str(cfg.model),
    )


def train_dataset(
    dataset_key: Dataset,
    data: LoadedDataset,
    n_splits: int = 5,
    repeats: int = 1,
    train_cfg: TrainConfig | None = None,
) -> list[ArticleResult]:
    """Every article-aligned variant of one cohort, from a single load."""
    out = []
    for cfg in article_variants(dataset_key):
        arm = "P1+P2" if cfg.protocol is None else f"P{cfg.protocol}"
        print(f"  {cfg.key.value} ({arm}, {'+'.join(cfg.modalities)}) ...", flush=True)
        res = train_regression_variant(
            cfg, data, n_splits=n_splits, repeats=repeats, train_cfg=train_cfg
        )
        verdict = "BEATS" if res.beats_baseline else "does not beat"
        print(f"    r={res.r_oof:+.3f} (p={res.p_perm:.3f}) "
              f"vs baseline BDI r={res.baseline_r_bdi_pre:+.3f} -> {verdict}",
              flush=True)
        out.append(res)
    return out


def article_table(results: list[ArticleResult]) -> str:
    head = (
        f"{'variant':<26}{'n':>5}{'feat':>6}"
        f"{'r (OOF)':>10}{'p_perm':>9}{'r_part':>8}{'r2':>8}"
        f"{'base BDI':>10}{'verdict':>10}"
    )
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(
            f"{r.key:<26}{r.n_patients:>5}{r.n_features:>6}"
            f"{r.r_oof:>+10.3f}{r.p_perm:>9.3f}{r.r_partial:>+8.3f}{r.r2:>+8.3f}"
            f"{r.baseline_r_bdi_pre:>+10.3f}"
            f"{('beats' if r.beats_baseline else '-'):>10}"
        )
    return "\n".join(lines)


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Train the article-aligned regression arm (per protocol)."
    )
    ap.add_argument("--root", type=Path, default=None,
                    help="TDBRAIN root (reads BDF); prefer --seed-db when seeded")
    ap.add_argument("--seed-db", type=Path, default=Path("recherche_tdbrain.sqlite3"),
                    help="rebuild the real cohort from this database instead of BDF")
    ap.add_argument("--sim-db", type=Path, default=Path("recherche_sim_matched.sqlite3"))
    ap.add_argument("--sim-only", action="store_true")
    ap.add_argument("--real-only", action="store_true")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1,
                    help="repetitions of the whole CV (the article used 10)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    from ..data.tdbrain_seeder import dataset_from_repository
    from ..db import Repository

    train_cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
    results: list[ArticleResult] = []

    if not args.real_only:
        print("matched simulated cohort (negative control) ...")
        sim = dataset_from_repository(Repository(db_url=f"sqlite:///{args.sim_db}"))
        results += train_dataset(
            Dataset.SIMULE, sim, args.n_splits, args.repeats, train_cfg
        )

    if not args.sim_only:
        print("\nreal TDBRAIN cohort ...")
        if args.root is not None:
            from ..data.tdbrain import TDBRAINConfig, load_tdbrain
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                real = load_tdbrain(TDBRAINConfig(
                    root=args.root, col_id="TDBRAIN_ID", col_protocol="rTMS PROTOCOL",
                ))
        else:
            real = dataset_from_repository(
                Repository(db_url=f"sqlite:///{args.seed_db}")
            )
        results += train_dataset(
            Dataset.TDBRAIN, real, args.n_splits, args.repeats, train_cfg
        )

    print("\n" + "=" * 92)
    print("ARTICLE-ALIGNED ARM - continuous target (delta BDI), protocols separated")
    print("=" * 92)
    print(article_table(results))
    print(
        f"\nr (OOF) = Pearson on out-of-fold predictions; p_perm = permutation "
        f"test ({N_PERMUTATIONS} draws).\n"
        "'base BDI' = correlation of baseline BDI-II alone with the target. "
        "A model that does not exceed it has added nothing beyond the intake form."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8"
    )
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    _main()
