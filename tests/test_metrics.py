"""Tests for the regression metrics, and for the trap one of them was built to avoid.

The article-aligned arm is scored by Pearson r on a target that is mathematically
coupled to baseline severity. Two things therefore have to be right, and both are
easy to get subtly wrong:

* the permutation p-value must behave like a p-value (near 1 for noise, at the
  floor for a strong signal), and
* the "did the EEG add anything beyond baseline severity?" statistic must not
  manufacture correlation out of a shared divisor. An earlier version did exactly
  that; `test_the_ratio_trap_is_not_reintroduced` pins the failure so the fix
  cannot be quietly reverted.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.metrics import (
    baseline_report,
    partial_correlation,
    pearson_r,
    permutation_p,
    r2,
    regression_report,
)


# --------------------------------------------------------------------------- #
# Pearson
# --------------------------------------------------------------------------- #
def test_pearson_matches_numpy_on_a_known_pair():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(50)
    b = 2.0 * a + rng.standard_normal(50) * 0.1
    assert pearson_r(a, b) == pytest.approx(np.corrcoef(a, b)[0, 1])


def test_pearson_is_nan_for_a_constant_prediction():
    """A model that learned nothing emits a constant; that is common, not exotic.

    Returning nan keeps it out of the fold means instead of raising mid-sweep.
    """
    a = np.arange(20, dtype=float)
    assert np.isnan(pearson_r(a, np.full(20, 3.0)))


def test_pearson_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="taille"):
        pearson_r(np.zeros(5), np.zeros(6))


# --------------------------------------------------------------------------- #
# Permutation test
# --------------------------------------------------------------------------- #
def test_permutation_p_is_at_the_floor_for_a_strong_signal():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(60)
    b = a + rng.standard_normal(60) * 0.05
    assert permutation_p(a, b, n_permutations=100, seed=0) == pytest.approx(1 / 101)


def test_permutation_p_is_unremarkable_for_noise():
    rng = np.random.default_rng(2)
    p = permutation_p(rng.standard_normal(60), rng.standard_normal(60),
                      n_permutations=100, seed=0)
    assert 0.05 < p <= 1.0


def test_permutation_p_can_never_be_zero():
    """The +1 estimator: n permutations cannot support a claim of p = 0."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal(40)
    assert permutation_p(a, a, n_permutations=10, seed=0) > 0.0


# --------------------------------------------------------------------------- #
# The ratio trap
# --------------------------------------------------------------------------- #
def test_the_ratio_trap_is_not_reintroduced():
    """Dividing both sides by a shared covariate invents correlation from nothing.

    A model emitting a *constant* prediction has learned exactly nothing. Scored
    by the ratio, it looked strongly predictive; scored by partial correlation, it
    correctly reads as noise. This test states both halves so the wrong statistic
    cannot come back as a "simplification".
    """
    rng = np.random.default_rng(0)
    n = 88
    bdi_pre = rng.uniform(14, 58, n)
    y_true = rng.uniform(0, 40, n)                 # independent of baseline
    y_pred = np.full(n, 17.0) + rng.standard_normal(n) * 0.01

    assert abs(pearson_r(y_true, y_pred)) < 0.15   # correctly ~0
    spurious = pearson_r(y_true / bdi_pre, y_pred / bdi_pre)
    assert spurious > 0.4                          # the trap, still reproducible
    assert abs(partial_correlation(y_true, y_pred, bdi_pre)) < 0.15


def test_partial_correlation_keeps_a_genuine_signal():
    rng = np.random.default_rng(4)
    n = 100
    covariate = rng.uniform(10, 50, n)
    y_true = 0.5 * covariate + rng.standard_normal(n) * 5
    y_pred = y_true + rng.standard_normal(n) * 1.0
    assert partial_correlation(y_true, y_pred, covariate) > 0.7


def test_partial_correlation_removes_a_covariate_only_effect():
    """When prediction and truth agree *only* through the covariate, nothing is left.

    This is the shape of the failure the column exists to catch: `bdi_pre` is one
    of the model's own input features, so a model can score a flattering raw r by
    re-emitting baseline severity and nothing else.
    """
    rng = np.random.default_rng(5)
    n = 120
    covariate = rng.uniform(10, 50, n)
    y_true = covariate + rng.standard_normal(n) * 2
    # Driven by the covariate, plus a little noise of its own.
    y_pred = covariate * 0.8 + 3.0 + rng.standard_normal(n) * 0.5

    assert pearson_r(y_true, y_pred) > 0.9                              # looks excellent...
    assert abs(partial_correlation(y_true, y_pred, covariate)) < 0.2    # ...adds nothing


def test_partial_correlation_is_nan_when_nothing_survives_residualising():
    """A prediction that is an *exact* function of the covariate leaves no residual.

    The partial correlation is then genuinely undefined (0/0), and nan says so.
    Callers treat a non-finite r as "did not beat the baseline", which is the
    right reading: such a model contributed nothing of its own.
    """
    rng = np.random.default_rng(8)
    covariate = rng.uniform(10, 50, 60)
    y_true = covariate + rng.standard_normal(60)
    assert np.isnan(partial_correlation(y_true, covariate * 0.8 + 3.0, covariate))


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def test_r2_is_negative_when_worse_than_the_mean():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, np.full(4, 100.0)) < 0


def test_regression_report_has_every_key_the_app_reads():
    rng = np.random.default_rng(6)
    a, b = rng.standard_normal(40), rng.standard_normal(40)
    report = regression_report(a, b, n_permutations=20)
    assert set(report) == {"r", "p_perm", "mae", "rmse", "r2", "n"}
    assert report["n"] == 40


def test_baseline_report_scores_a_covariate_with_no_model():
    """The bar an EEG model must clear, computed straight from the intake form."""
    rng = np.random.default_rng(7)
    bdi_pre = rng.uniform(14, 58, 60)
    delta = 0.5 * bdi_pre + rng.standard_normal(60) * 3
    report = baseline_report(delta, bdi_pre, n_permutations=50)
    assert report["r"] > 0.7
    assert report["p_perm"] < 0.05
