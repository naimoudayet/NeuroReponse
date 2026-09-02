"""How much is one Pearson r worth on 44 patients? Measured, not assumed.

The reference study (Arteaga et al., PMC12981298) reports a single r per protocol
— r = 0.401 on protocol 1 (n = 44), r = 0.26 on protocol 2 — as its headline
result, with a permutation p attached. This project cannot compare its own r to
that number honestly without first knowing **how much an r of that kind moves
when nothing about the data changes**.

    python -m src.reporting.r_stability

Two distributions are measured on the *same* cohort, the *same* protocol-1 arm
and the *same* EEG-only feature set the study uses:

``real``
    The full pipeline retrained across seeds. Only the fold assignment and the
    weight initialisation change; the patients, features and target are fixed.
    Any spread here is spread the article's single number also has, and does not
    report.

``shuffled``
    The same, with the target permuted across patients before anything is fitted.
    The true correlation is zero by construction, so this is the null the r must
    clear — and, unlike a permutation test applied after prediction, it retrains
    end to end, so it also catches a leak. This project has direct experience of
    that distinction: a leaked run reported p = 0.010 on numbers that were
    entirely artefactual.

The comparison that matters is whether the two distributions overlap. If they do,
a single r on this cohort — ours *or* anyone's — is not evidence.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..data.loader import LoadedDataset
from ..data.modalities import build_features, protocol_mask, target_values
from ..models.lstm import REGRESSION, LSTMConfig
from ..models.metrics import pearson_r
from ..models.train import TrainConfig, cross_validate

OUT_PATH = Path("data/models/r_stability.json")

# The article's reported correlations, by protocol. Kept here so the comparison
# lives in the artefact rather than in a reader's memory.
ARTICLE_R: dict[int, float] = {1: 0.401, 2: 0.26}


@dataclass
class Stability:
    protocol: int
    modalities: list[str]
    target: str
    n_patients: int
    n_features: int
    n_runs: int
    article_r: float
    baseline_r_bdi_pre: float
    # Bootstrap interval on the trivial baseline itself. Load-bearing: on 44
    # patients the confound's own r carries a CI wide enough to contain the
    # article's headline number, so "0.500 beats 0.401" is a point-estimate
    # statement that does not survive its own uncertainty. The defensible claim
    # is the weaker and more damaging one — the two are indistinguishable, and a
    # model that cannot be told apart from a variable requiring no model has not
    # demonstrated added value.
    baseline_ci_lo: float
    baseline_ci_hi: float
    baseline_exceeds_article_frac: float
    real_r: list[float]
    shuffled_r: list[float]
    seconds: float

    # ------------------------------------------------------------------ #
    # Summaries. Computed rather than stored so a reader of the JSON can
    # recheck them against the raw lists, which are kept in full.
    # ------------------------------------------------------------------ #
    @property
    def real_mean(self) -> float:
        return float(np.mean(self.real_r))

    @property
    def real_sd(self) -> float:
        return float(np.std(self.real_r))

    @property
    def null_sd(self) -> float:
        return float(np.std(self.shuffled_r))

    @property
    def null_max_abs(self) -> float:
        return float(np.max(np.abs(self.shuffled_r)))

    def overlap(self) -> float:
        """Fraction of shuffled runs reaching at least the mean real r.

        A high value means the two distributions sit on top of each other: the
        pipeline does as well on randomised labels as on the real ones.
        """
        return float(np.mean(np.asarray(self.shuffled_r) >= self.real_mean))

    def article_r_percentile_in_null(self) -> float:
        """Where the article's r would fall inside *this* cohort's null.

        Not a test of their result — their pipeline is different and stronger
        (ICA, itEMD, SBLEST spatial filters). It answers a narrower and still
        useful question: on a cohort of this size, how unusual is a correlation
        of that magnitude when the labels carry no information at all?
        """
        return float(np.mean(np.asarray(self.shuffled_r) < self.article_r) * 100.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({
            "real_mean": self.real_mean,
            "real_sd": self.real_sd,
            "null_sd": self.null_sd,
            "null_max_abs": self.null_max_abs,
            "overlap": self.overlap(),
            "article_r_percentile_in_null": self.article_r_percentile_in_null(),
        })
        return d


def baseline_interval(
    y: np.ndarray,
    covariate: np.ndarray,
    article_r: float,
    n_boot: int = 5000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``(ci_lo, ci_hi, fraction of resamples above ``article_r``)``.

    The baseline here is a *correlation of a raw covariate with the target* — no
    model, no folds — so it has no training variance at all. What it does have is
    sampling variance, and on 44 patients that is large. Reporting the point
    estimate alone would let a comparison rest on a difference the sample cannot
    resolve.
    """
    a = np.asarray(covariate, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, a.size, a.size)
        r = pearson_r(b[idx], a[idx])
        if np.isfinite(r):
            draws.append(r)
    if not draws:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(np.asarray(draws) > article_r))


def _one_run(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    n_splits: int,
    epochs: int,
    standardise: bool,
) -> float:
    """One complete retrain + cross-validation; returns the pooled out-of-fold r."""
    cv = cross_validate(
        x, y.astype(np.float32), groups,
        lstm_cfg=LSTMConfig(input_size=x.shape[-1], task=REGRESSION),
        train_cfg=TrainConfig(epochs=epochs, seed=seed),
        n_splits=n_splits, standardise=standardise,
    )
    y_true, y_pred = cv.out_of_fold(y)
    return float(pearson_r(y_true, y_pred))


def measure(
    dataset: LoadedDataset,
    protocol: int = 1,
    modalities: tuple[str, ...] = ("eeg",),
    target: str = "delta_bdi",
    n_runs: int = 15,
    n_splits: int = 5,
    epochs: int = 30,
    standardise: bool = True,
    verbose: bool = True,
) -> Stability:
    """Both distributions, from one feature build."""
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, _, groups, _ = build_features(
            dataset, modalities=modalities, target=target, protocol=protocol
        )
    mask = protocol_mask(dataset, protocol)
    y = target_values(dataset, target)[mask]
    bdi_pre = dataset.metadata["bdi_pre"].to_numpy(dtype=np.float64)[mask]

    real: list[float] = []
    shuffled: list[float] = []
    for k in range(n_runs):
        real.append(_one_run(x, y, groups, k, n_splits, epochs, standardise))
        # Permute the target across patients. Groups are one row per patient
        # here, so a plain permutation already respects the group structure.
        rng = np.random.default_rng(10_000 + k)
        shuffled.append(
            _one_run(x, rng.permutation(y), groups, k, n_splits, epochs, standardise)
        )
        if verbose:
            print(f"  run {k + 1:>2}/{n_runs}  réel r={real[-1]:+.3f}   "
                  f"permuté r={shuffled[-1]:+.3f}", flush=True)

    article_r = ARTICLE_R.get(protocol, float("nan"))
    ci_lo, ci_hi, above = baseline_interval(y, bdi_pre, article_r)
    return Stability(
        protocol=protocol, modalities=list(modalities), target=target,
        n_patients=int(x.shape[0]), n_features=int(x.shape[-1]), n_runs=n_runs,
        article_r=article_r,
        baseline_r_bdi_pre=float(pearson_r(y, bdi_pre)),
        baseline_ci_lo=ci_lo, baseline_ci_hi=ci_hi,
        baseline_exceeds_article_frac=above,
        real_r=real, shuffled_r=shuffled, seconds=float(time.time() - t0),
    )


def report(s: Stability) -> str:
    lines = [
        f"Protocole {s.protocol} · {'+'.join(s.modalities)} · cible {s.target} · "
        f"n = {s.n_patients} patients, {s.n_features} variables, {s.n_runs} "
        f"réentraînements complets",
        "",
        f"  r réel        moyenne {s.real_mean:+.3f}   écart-type {s.real_sd:.3f}   "
        f"étendue [{min(s.real_r):+.3f}, {max(s.real_r):+.3f}]",
        f"  r permuté     moyenne {np.mean(s.shuffled_r):+.3f}   "
        f"écart-type {s.null_sd:.3f}   "
        f"étendue [{min(s.shuffled_r):+.3f}, {max(s.shuffled_r):+.3f}]",
        "",
        f"  Recouvrement : {s.overlap():.0%} des runs sur étiquettes permutées "
        f"atteignent au moins le r réel moyen.",
        f"  |r| maximal atteint SANS aucune information dans les étiquettes : "
        f"{s.null_max_abs:.3f}",
        f"  Référence triviale (BDI-II initial seul, sans modèle) : "
        f"r = {s.baseline_r_bdi_pre:+.3f}  "
        f"IC95 [{s.baseline_ci_lo:+.3f}, {s.baseline_ci_hi:+.3f}]",
        f"  Article (protocole {s.protocol}) : r = {s.article_r:.3f} — "
        f"{s.article_r_percentile_in_null():.0f}e centile de ce bruit ; "
        f"{s.baseline_exceeds_article_frac:.0%} des rééchantillons de la "
        f"référence triviale le dépassent.",
    ]
    return "\n".join(lines)


def read_json(path: Path = OUT_PATH) -> list[dict]:
    """The measurement as the app reads it; empty when it has never been run."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure how much a single Pearson r moves on this cohort."
    )
    ap.add_argument("--db", type=Path, default=Path("recherche_tdbrain.sqlite3"))
    ap.add_argument("--protocols", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    from ..data.tdbrain_seeder import dataset_from_repository
    from ..db import Repository

    print(f"loading {args.db} ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = dataset_from_repository(Repository(db_url=f"sqlite:///{args.db}"))

    results: list[Stability] = []
    for protocol in args.protocols:
        print(f"\nprotocole {protocol} — EEG seul (le montage de l'article) ...")
        s = measure(
            dataset, protocol=protocol, n_runs=args.runs,
            n_splits=args.n_splits, epochs=args.epochs,
        )
        print("\n" + report(s))
        results.append(s)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps([s.to_dict() for s in results], indent=2), encoding="utf-8"
    )
    print(f"\nrésultats -> {args.out}")


if __name__ == "__main__":
    _main()
