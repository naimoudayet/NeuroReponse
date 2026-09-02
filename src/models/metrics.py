"""Regression metrics, reported the way the reference study reports them.

Arteaga et al. (PMC12981298) predict a **continuous** outcome — the change in
BDI-II from before to after the rTMS course — and score it with Pearson's *r*
against a permutation null. This project's existing metrics are all
classification (AUC, accuracy, F1), which cannot be compared with theirs at all,
so the regression side lives here.

Two things this module insists on, because both are easy to get wrong on a
cohort of 44 patients:

* **A permutation p-value, not the analytic one.** ``scipy.stats.pearsonr``'s
  p assumes bivariate normality and independent observations. With cross-
  validated predictions neither holds — the folds share a model class, the
  target is bounded below by ``-bdi_pre``, and n is small. Shuffling the labels
  and re-scoring is the assumption-free answer, and it is what the article did
  (100 iterations).
* **A trivial baseline alongside every number.** ``delta_bdi`` is mathematically
  coupled to baseline severity: you cannot drop 40 points from a BDI of 20.
  Measured on this cohort, ``bdi_pre`` alone reaches r = 0.500 on protocol 1 —
  *the same magnitude as the article's headline r = 0.401 from EEG*, and with a
  bootstrap CI of [0.258, 0.700] the two cannot be told apart. An r reported without
  that comparison is not interpretable, so :func:`regression_report` carries the
  baseline in the same dict rather than leaving it to the caller to remember.
"""
from __future__ import annotations

import numpy as np

# The article's permutation count. Coarse (the smallest reachable p is
# 1/(n+1) ~ 0.0099) but it is what the comparison has to match.
N_PERMUTATIONS = 100


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation, ``nan`` when either side is constant.

    A constant prediction is exactly what a model that has learned nothing
    produces, so this case is common rather than exotic — returning ``nan``
    keeps it out of the means instead of raising mid-sweep.
    """
    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_pred, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"taille incompatible : {a.size} vs {b.size}")
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def permutation_p(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 0,
) -> float:
    """One-sided permutation p for ``pearson_r``: how often chance does as well.

    Shuffling ``y_true`` breaks any true association while preserving both
    marginal distributions, so the null keeps the target's skew and the
    prediction's shape. One-sided because a *negative* correlation between
    predicted and observed improvement is not a weaker success, it is a
    different (and worse) failure than noise.

    The ``+1`` on both sides is the standard unbiased estimator — it counts the
    observed statistic as one draw from its own null, so the result can never be
    exactly 0, which would claim more certainty than n permutations can support.
    """
    observed = pearson_r(y_true, y_pred)
    if not np.isfinite(observed):
        return float("nan")

    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_pred, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    n_ge = sum(
        1 for _ in range(n_permutations)
        if pearson_r(rng.permutation(a), b) >= observed
    )
    return float((n_ge + 1) / (n_permutations + 1))


def _residualise(v: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """``v`` with its best linear fit on ``covariate`` removed."""
    c = np.asarray(covariate, dtype=np.float64).ravel()
    a = np.asarray(v, dtype=np.float64).ravel()
    design = np.column_stack([np.ones_like(c), c])
    coef, *_ = np.linalg.lstsq(design, a, rcond=None)
    return a - design @ coef


def partial_correlation(
    y_true: np.ndarray, y_pred: np.ndarray, covariate: np.ndarray
) -> float:
    """Correlation between prediction and truth **after removing ``covariate``**.

    This answers "did the model find anything that baseline severity did not
    already contain?" — the question the raw r cannot answer when the target is
    mathematically coupled to that baseline.

    It replaces an earlier, wrong attempt at the same idea: dividing both truth
    and prediction by ``bdi_pre`` and correlating the ratios. Sharing a divisor
    manufactures correlation (Pearson's spurious-ratio problem). Measured on this
    project's own cohort shape, a model emitting a **constant** prediction — one
    that has learned nothing whatsoever — scored r = 0.539 that way. Residualising
    both sides on the covariate instead is the standard fix, and it correctly
    returns ~0 for that same constant model, because a constant has no variance
    left once the covariate is projected out.
    """
    a = _residualise(y_true, covariate)
    b = _residualise(y_pred, covariate)
    return pearson_r(a, b)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.sqrt(np.mean(d ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination. Negative means worse than the mean.

    Reported precisely because it *can* go negative: a model that regresses to
    an unhelpful constant scores r2 <= 0 while still producing a flattering
    ``mae``, and the pair together tells you which happened.
    """
    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 0,
) -> dict[str, float]:
    """Every regression number this project reports, in one dict."""
    return {
        "r": pearson_r(y_true, y_pred),
        "p_perm": permutation_p(y_true, y_pred, n_permutations, seed),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "n": int(np.asarray(y_true).size),
    }


def baseline_report(
    y_true: np.ndarray,
    covariate: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 0,
) -> dict[str, float]:
    """The bar a neurophysiological model has to clear.

    ``covariate`` is a single clinical variable — baseline BDI-II or age — used
    raw, with no model fitted. If the EEG model's r does not exceed this, the
    EEG contributed nothing that the intake form did not already contain.
    """
    return regression_report(y_true, covariate, n_permutations, seed)


# --------------------------------------------------------------------------- #
# Classification — the reporting rule from the guide's section 5.9 / 5.10
# --------------------------------------------------------------------------- #
#
# The project's classification reporting was accuracy / ROC-AUC / F1. On a cohort
# whose base rate is 83/132 = 0.629 that trio is close to unreadable: an
# all-positive predictor scores accuracy 0.629 and F1 0.768, both of which look
# like a working model. Three of the metrics added below make that failure
# visible instead:
#
#   * **balanced accuracy** is 0.500 for the all-positive predictor, by
#     construction, whatever the base rate.
#   * **specificity** is 0.000 for it — the empty column of the confusion matrix,
#     as a number.
#   * **Brier** scores the probabilities rather than the thresholded labels, so a
#     model that is confidently wrong is separated from one that is unsure.
#
# PR-AUC is added because ROC-AUC is insensitive to class imbalance in the
# direction that flatters the majority class, and its own no-skill value is the
# base rate rather than 0.5 — which is why `classification_report_full` returns
# `pr_auc_baseline` next to it. A PR-AUC of 0.63 on this cohort is *nothing*.

def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    t = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_pred).astype(int).ravel()
    return (
        int(np.sum((t == 1) & (p == 1))),   # tp
        int(np.sum((t == 0) & (p == 0))),   # tn
        int(np.sum((t == 0) & (p == 1))),   # fp
        int(np.sum((t == 1) & (p == 0))),   # fn
    )


def brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Mean squared error of the probabilities: the guide's calibration term.

    Lower is better, and the reference point is not 0 — a model that always
    emits the base rate p scores p(1-p), which is 0.234 on this cohort. A Brier
    above that is worse than knowing nothing but the base rate.
    """
    t = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_proba, dtype=np.float64).ravel()
    return float(np.mean((p - t) ** 2))


def permutation_p_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 0,
) -> float:
    """One-sided permutation p for ROC-AUC — how often chance separates as well.

    Same caveat as :func:`permutation_p`: this tests the *statistic*, not the
    pipeline. It shuffles labels after the predictions already exist, so it
    cannot detect a leak upstream of them. This project has direct experience of
    that — the early-stopping leak produced p = 0.010 on numbers that were
    entirely artefactual. The decisive test retrains on permuted targets.
    """
    from sklearn.metrics import roc_auc_score

    t = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_proba, dtype=np.float64).ravel()
    if t.size < 2 or len(np.unique(t)) < 2:
        return float("nan")
    observed = float(roc_auc_score(t, p))
    rng = np.random.default_rng(seed)
    n_ge = sum(
        1 for _ in range(n_permutations)
        if float(roc_auc_score(rng.permutation(t), p)) >= observed
    )
    return float((n_ge + 1) / (n_permutations + 1))


def bootstrap_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    statistic="auc",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval over patients, for AUC or balanced accuracy.

    Resamples **patients**, which is the independent unit here — resampling rows
    would treat a patient's eight epochs as eight observations and shrink every
    interval by roughly sqrt(8). Draws that end up single-class are skipped
    rather than scored, since AUC is undefined on them.
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    t = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_proba, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, t.size, t.size)
        tt, pp = t[idx], p[idx]
        if len(np.unique(tt)) < 2:
            continue
        vals.append(
            float(roc_auc_score(tt, pp)) if statistic == "auc"
            else float(balanced_accuracy_score(tt, (pp >= 0.5).astype(int)))
        )
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def classification_report_full(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    n_permutations: int = N_PERMUTATIONS,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Every classification number the guide's section 5.9 asks for, in one dict.

    Includes the two no-skill reference points (``base_rate`` and
    ``brier_baseline``) so no caller has to remember them: on an imbalanced
    cohort, accuracy and PR-AUC are meaningless without the base rate printed
    beside them, and this is the project whose four-variant table reads as four
    working models until you notice all four match it exactly.
    """
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    t = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_proba, dtype=np.float64).ravel()
    pred = (p >= threshold).astype(int)
    tp, tn, fp, fn = _confusion(t, pred)
    two_class = len(np.unique(t)) >= 2
    base_rate = float(t.mean()) if t.size else float("nan")

    return {
        "n": int(t.size),
        "base_rate": base_rate,
        "accuracy": float((tp + tn) / max(t.size, 1)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(t, pred)) if two_class else float("nan")
        ),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "f1": float(f1_score(t, pred, zero_division=0)),
        "auc": float(roc_auc_score(t, p)) if two_class else float("nan"),
        "pr_auc": float(average_precision_score(t, p)) if two_class else float("nan"),
        "pr_auc_baseline": base_rate,          # no-skill PR-AUC *is* the base rate
        "brier": brier(t, p),
        "brier_baseline": float(base_rate * (1.0 - base_rate)),
        "p_perm_auc": permutation_p_auc(t, p, n_permutations, seed),
        "auc_ci_lo": bootstrap_ci(t, p, "auc", n_boot, seed=seed)[0],
        "auc_ci_hi": bootstrap_ci(t, p, "auc", n_boot, seed=seed)[1],
        # The single number that says "this is the majority-class predictor":
        # 0 means the model never used one of the two classes.
        "predicted_positive_rate": float(pred.mean()) if pred.size else float("nan"),
    }


def benjamini_hochberg(p_values, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """FDR-adjusted q-values and rejection flags, ``(rejected, q)``.

    HYPO4-369 makes multiple-comparison correction part of its own falsification
    criterion — "si l'ajout 3-6-9 ... disparaît après correction des
    comparaisons multiples, H369 doit être considérée comme non soutenue" — and
    the univariate screen it implies runs 40 correlations at once, where roughly
    two hits below p = 0.05 are expected from noise alone.

    Implemented here rather than pulled from statsmodels: it is six lines, and
    this project requires an upper bound on every dependency, so a new one has to
    earn its place.

    Non-finite p-values are carried through as ``nan`` and never rejected, which
    is what a constant feature produces.
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    finite = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    if not finite.any():
        return np.zeros(p.shape, dtype=bool), q
    sub = p[finite]
    order = np.argsort(sub)
    n = sub.size
    ranked = sub[order] * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downwards, the standard BH step-up.
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    q[finite] = out
    return (np.nan_to_num(q, nan=1.0) <= alpha) & finite, q


def beats_chance(report: dict[str, float], alpha: float = 0.05) -> bool:
    """The guide's section 5.10 selection rule, as one boolean.

    A model counts only if **all four** hold: its AUC confidence interval
    excludes 0.5, its permutation p clears ``alpha``, its balanced accuracy
    exceeds 0.5, and it actually predicted both classes. Ranking on accuracy is
    explicitly not part of it — on this cohort the majority-class predictor wins
    that ranking, which is how a null result gets published as a positive one.
    """
    return bool(
        np.isfinite(report.get("auc_ci_lo", np.nan))
        and report["auc_ci_lo"] > 0.5
        and np.isfinite(report.get("p_perm_auc", np.nan))
        and report["p_perm_auc"] <= alpha
        and report.get("balanced_accuracy", 0.0) > 0.5
        and 0.0 < report.get("predicted_positive_rate", 0.0) < 1.0
    )
