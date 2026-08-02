"""Tests for the positive control, and for the trap it exposed.

The sweep exists to prove the pipeline can detect an effect when one is present.
Measuring it turned up something sharper: the simulator raises responders' alpha
by a **per-patient constant**, and per-patient z-scoring centres each patient on
their own epochs — so the default preprocessing subtracts exactly the quantity
that carries the label.

That is why the raw arm exists, and why these tests pin the direction: a change
that made the raw arm stop recovering the effect, or made the z-scored arm start
recovering it, would mean the simulator or the normalisation had changed
meaning.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.modalities import build_features
from src.data.simulator_matched import MatchedSimConfig, simulate_matched
from src.reporting.effect_sweep import SweepPoint, figure_effect_curve, run_point, write_csv

# Small enough to train quickly, large enough for 2-fold patient-wise CV.
TINY = MatchedSimConfig(n_patients=40, n_epochs=4, window=500, n_rr=32)
STRONG = 0.8


def _auc(effect: float, zscore: bool) -> float:
    return run_point(
        effect, ("eeg",), zscore=zscore,
        n_splits=2, epochs=8, seed=3, sim=TINY,
    ).auc_mean


def test_zscoring_removes_the_between_patient_effect():
    """The mechanism, measured on the features rather than through a model.

    Cheaper and far more direct than an AUC comparison: after z-scoring, the
    responder/non-responder difference in the affected band must collapse.
    """
    dataset = simulate_matched(
        MatchedSimConfig(**{**TINY.__dict__, "effect_size": STRONG})
    )
    labels = dataset.labels.astype(bool)

    raw, _y, _g, _n = build_features(dataset, modalities=("eeg",), per_patient_zscore=False)
    zed, _y, _g, _n = build_features(dataset, modalities=("eeg",), per_patient_zscore=True)

    def separation(x):
        """Largest standardised gap between the two groups, over all columns."""
        per_patient = x.mean(axis=1)
        a, b = per_patient[labels], per_patient[~labels]
        pooled = np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-12
        return float(np.max(np.abs(a.mean(0) - b.mean(0)) / pooled))

    assert separation(raw) > 0.5, "the simulator injected no detectable effect"
    assert separation(zed) < separation(raw) / 2, (
        "per-patient z-scoring should erase a per-patient constant shift"
    )


@pytest.mark.slow
def test_raw_arm_recovers_the_effect_that_the_zscored_arm_loses():
    """End-to-end version of the same claim, through the actual model."""
    assert _auc(STRONG, zscore=False) > _auc(STRONG, zscore=True)


def test_sweep_point_is_serialisable_and_plottable(tmp_path):
    points = [
        SweepPoint(features="eeg", zscore=z, effect_size=e, auc_mean=a,
                   auc_std=0.1, accuracy_mean=0.6, f1_mean=0.7, base_rate=0.63,
                   n_patients=40, n_features=130)
        for z, offset in ((False, 0.35), (True, 0.0))
        for e, a in zip((0.0, 0.5), (0.5, 0.5 + offset))
    ]
    csv_path = write_csv(points, tmp_path / "sweep.csv")
    assert "zscore" in csv_path.read_text(encoding="utf-8")

    fig_path = figure_effect_curve(points, tmp_path)
    assert fig_path.exists() and fig_path.stat().st_size > 0
