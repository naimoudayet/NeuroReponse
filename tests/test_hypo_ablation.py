"""Tests for the HYPO4 / RES0_AR1 ablation ladder and the rules it is scored by.

Three things here are load-bearing rather than decorative:

* the ladder assembles cached blocks itself, so it must produce a vector
  **identical** to the single-call ``build_features`` path — otherwise every rung
  is silently a different experiment from the one the rest of the project runs;
* ``beats_chance`` must reject the all-positive predictor, which is the exact
  failure mode this cohort produces and which accuracy and F1 both reward;
* ``physics_proxy`` must be provably collinear with the protocol, since the claim
  "Model E is not merely unimplemented, it is empty" rests on that and nothing
  else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import LoadedDataset
from src.data.modalities import MODALITY_ORDER, build_features
from src.models.metrics import beats_chance, benjamini_hochberg, brier
from src.models.train import TrainConfig, apply_standardiser, fit_standardiser
from src.reporting.hypo_ablation import (
    LADDER,
    REG_LADDER,
    assemble,
    build_all_blocks,
    feature_validity_report,
    physics_is_collinear,
    physics_proxy,
    run_ladder,
)

CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4",
    "T7", "C3", "Cz", "C4", "T8", "CP3", "CPz", "CP4", "P7", "P3",
    "Pz", "P4", "P8", "O1", "Oz", "O2",
]


def _dataset(n_patients: int = 16, n_epochs: int = 3, window: int = 750) -> LoadedDataset:
    """A TDBRAIN-shaped cohort, small enough to train in a test."""
    rng = np.random.default_rng(0)
    mc = rng.standard_normal((n_patients, n_epochs, len(CHANNELS), window)).astype(np.float32)
    bdi_pre = rng.uniform(20.0, 45.0, n_patients)
    bdi_post = rng.uniform(5.0, 40.0, n_patients)
    return LoadedDataset(
        signals=mc[:, :, CHANNELS.index("Pz"), :],
        labels=((bdi_pre - bdi_post) / bdi_pre >= 0.5).astype(int),
        fs=250.0,
        window=window,
        metadata=pd.DataFrame({
            "patient_id": [f"P{i:03d}" for i in range(n_patients)],
            "protocol": rng.integers(1, 3, n_patients),
            "age": rng.uniform(20.0, 70.0, n_patients),
            "gender": rng.integers(0, 2, n_patients),
            "bdi_pre": bdi_pre,
            "bdi_post": bdi_post,
            "pct_reduction": (bdi_pre - bdi_post) / bdi_pre,
        }),
        ecg=rng.uniform(0.7, 1.1, (n_patients, n_epochs, 64)).astype(np.float32),
        channels=list(CHANNELS),
        signals_mc=mc,
    )


# --------------------------------------------------------------------------- #
# The equality that makes the ladder an experiment rather than eleven of them
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("modalities", [m for _, _, m in LADDER])
def test_ladder_assembly_matches_build_features(modalities):
    """Cached-block assembly must equal the single-call path, exactly.

    ``build_all_blocks`` exists only to avoid recomputing the 30 synchronisation
    features once per rung. The moment its concatenation order drifts from
    ``MODALITY_ORDER``, every rung trains on a permuted vector and the ladder
    stops comparing what it says it compares — with no error anywhere.
    """
    dataset = _dataset()
    blocks = build_all_blocks(dataset, verbose=False)
    cached, cached_names = assemble(blocks, modalities)
    direct, _, _, direct_names = build_features(dataset, modalities=modalities)
    assert cached.shape == direct.shape
    assert cached_names == direct_names
    assert np.max(np.abs(cached - direct)) == 0.0


def test_assembly_ignores_the_order_the_caller_spells_modalities_in():
    dataset = _dataset()
    blocks = build_all_blocks(dataset, verbose=False)
    a, names_a = assemble(blocks, ("h369", "rtms", "sync"))
    b, names_b = assemble(blocks, ("rtms", "sync", "h369"))
    assert names_a == names_b
    assert np.array_equal(a, b)


def test_every_ladder_rung_names_only_known_modalities():
    for _, _, modalities in LADDER + REG_LADDER:
        assert set(modalities) <= set(MODALITY_ORDER)


def test_connectivity_refuses_a_single_channel_dataset():
    """PLV between one channel and itself is 1.0, and would pass every shape check."""
    from src.data.modalities import ModalityError

    dataset = _dataset()
    dataset.signals_mc = dataset.signals_mc[:, :, :1, :]
    dataset.channels = CHANNELS[:1]
    with pytest.raises(ModalityError, match="multi-channel"):
        build_features(dataset, modalities=("sync",))


# --------------------------------------------------------------------------- #
# Model E — the electromagnetic layer
# --------------------------------------------------------------------------- #

def test_physics_proxy_adds_no_rank_over_the_protocol():
    """The numerical form of "Model E is empty on this dataset".

    Every B/E/J quantity TDBRAIN's metadata can support is a function of the
    protocol integer, because the coil current, geometry and tissue conductivity
    are all unpublished. Equal ranks is what that means arithmetically.
    """
    dataset = _dataset()
    result = physics_is_collinear(dataset)
    assert result["n_physics_columns"] == 3
    assert result["rank_protocol"] == result["rank_protocol_plus_physics"]


def test_physics_proxy_is_a_deterministic_function_of_the_protocol():
    dataset = _dataset()
    phys, names = physics_proxy(dataset)
    protocol = dataset.metadata["protocol"].to_numpy()
    assert len(names) == phys.shape[1]
    for value in np.unique(protocol):
        rows = phys[protocol == value]
        assert np.allclose(rows, rows[0]), "physics varies within one protocol arm"


# --------------------------------------------------------------------------- #
# The stopping rule
# --------------------------------------------------------------------------- #

def test_beats_chance_rejects_the_all_positive_predictor():
    """The failure this project actually produces, and the rule that catches it.

    An 83/132 base rate makes the constant "responder" answer score accuracy
    0.629 and F1 0.768 — both of which read as a working model. Balanced accuracy
    is 0.500 for it and specificity 0.000, which is why the rule is written on
    those and not on accuracy.
    """
    from src.models.metrics import classification_report_full

    y = np.array([1] * 83 + [0] * 49)
    report = classification_report_full(
        y, np.full(y.size, 0.8), n_permutations=20, n_boot=100
    )
    assert report["accuracy"] == pytest.approx(83 / 132, abs=1e-6)
    assert report["f1"] > 0.75                      # looks like a working model
    assert report["balanced_accuracy"] == pytest.approx(0.5)
    assert report["specificity"] == 0.0
    assert report["predicted_positive_rate"] == 1.0
    assert not beats_chance(report)


def test_beats_chance_accepts_a_genuine_separation():
    from src.models.metrics import classification_report_full

    rng = np.random.default_rng(1)
    y = (rng.random(160) < 0.6).astype(int)
    proba = np.clip(0.5 + 0.35 * (y - 0.5) + rng.normal(0, 0.10, y.size), 0.0, 1.0)
    report = classification_report_full(y, proba, n_permutations=100, n_boot=300)
    assert beats_chance(report)


def test_pr_auc_baseline_is_the_base_rate_not_one_half():
    """A PR-AUC of 0.63 on this cohort is nothing, and the report must say so."""
    from src.models.metrics import classification_report_full

    y = np.array([1] * 83 + [0] * 49)
    report = classification_report_full(
        y, np.full(y.size, 0.8), n_permutations=10, n_boot=50
    )
    assert report["pr_auc"] == pytest.approx(report["pr_auc_baseline"], abs=0.02)
    assert report["pr_auc_baseline"] == pytest.approx(83 / 132)


def test_brier_baseline_is_the_score_of_predicting_the_base_rate():
    y = np.array([1] * 83 + [0] * 49)
    p = float(y.mean())
    assert brier(y, np.full(y.size, p)) == pytest.approx(p * (1 - p), abs=1e-9)


# --------------------------------------------------------------------------- #
# Multiple comparisons
# --------------------------------------------------------------------------- #

def test_benjamini_hochberg_matches_a_hand_computed_case():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    rejected, q = benjamini_hochberg(p, alpha=0.05)
    # Largest k with p(k) <= k*alpha/n is k = 2 (0.008 <= 0.010; 0.039 > 0.015).
    assert int(rejected.sum()) == 2
    assert q[0] == pytest.approx(0.01)
    assert q[1] == pytest.approx(0.04)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)   # monotone after sorting


def test_benjamini_hochberg_is_stricter_than_the_raw_threshold():
    """Forty tests at alpha = 0.05 yield ~2 nominal hits from noise alone.

    This is why the 3-6-9 screen is corrected: the ladder tests 40 features
    against three outcomes, and an uncorrected read would find "signal" every
    time.
    """
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 40)
    rejected, _ = benjamini_hochberg(p, alpha=0.05)
    assert int(rejected.sum()) <= int((p < 0.05).sum())


# --------------------------------------------------------------------------- #
# Standardisation
# --------------------------------------------------------------------------- #

def test_standardiser_preserves_between_patient_differences():
    """The property that separates it from ``zscore_epochs``, which does not.

    Cohort standardisation only changes the units; per-patient z-scoring removes
    each patient's own mean, which is where a between-patient label lives. Two
    patients that differ must still differ afterwards.
    """
    x = np.zeros((2, 4, 1), dtype=np.float32)
    x[0] = 1.0
    x[1] = 5.0
    z = apply_standardiser(x, fit_standardiser(x))
    assert z[0].mean() != pytest.approx(z[1].mean())

    from src.data.tdbrain import zscore_epochs
    collapsed = zscore_epochs(x)
    assert collapsed[0].mean() == pytest.approx(collapsed[1].mean())


def test_standardiser_survives_a_constant_column():
    x = np.ones((3, 2, 4), dtype=np.float32)
    z = apply_standardiser(x, fit_standardiser(x))
    assert np.isfinite(z).all()


def test_cross_validate_standardise_defaults_to_off():
    """Every figure recorded before this existed must still reproduce exactly."""
    import inspect

    from src.models.train import cross_validate

    assert inspect.signature(cross_validate).parameters["standardise"].default is False


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

def test_ladder_runs_and_reports_every_rung():
    dataset = _dataset(n_patients=18, n_epochs=2, window=750)
    rows = run_ladder(
        dataset, n_splits=3, train_cfg=TrainConfig(epochs=1), verbose=False
    )
    assert len(rows) == len(LADDER)
    for row in rows:
        assert row.n_features > 0
        assert np.isfinite(row.brier)
        assert row.pr_auc_baseline == pytest.approx(row.base_rate)


def test_feature_validity_report_screens_every_new_feature():
    dataset = _dataset(n_patients=20, n_epochs=2, window=750)
    report = feature_validity_report(dataset)
    screen = report["screens"]["responder"]
    # 30 sync + 6 complexity + 4 harmonic ratios.
    assert screen["n_tested"] == 40
    assert screen["n_survivors_fdr"] <= screen["n_nominal_hits"]
    assert len(report["age_controls"]) == 3


# --------------------------------------------------------------------------- #
# How much is one r worth on this cohort?
# --------------------------------------------------------------------------- #

def test_baseline_interval_brackets_its_point_estimate():
    """The bootstrap CI must contain the statistic it is an interval for."""
    from src.models.metrics import pearson_r
    from src.reporting.r_stability import baseline_interval

    rng = np.random.default_rng(3)
    x = rng.normal(30.0, 8.0, 44)
    y = 0.5 * x + rng.normal(0.0, 8.0, 44)
    point = pearson_r(y, x)
    lo, hi, above = baseline_interval(y, x, article_r=0.401, n_boot=800)
    assert lo < point < hi
    assert 0.0 <= above <= 1.0


def test_baseline_interval_is_wide_enough_to_matter_at_n_44():
    """The reason the "0.500 beats 0.401" claim had to be softened.

    On 44 patients a correlation near 0.5 carries an interval roughly 0.4 wide.
    Any head-to-head that rests on a 0.1 gap between two such numbers is reading
    sampling noise, so this pins that the interval is reported and is not
    accidentally collapsing to a point.
    """
    from src.reporting.r_stability import baseline_interval

    rng = np.random.default_rng(4)
    x = rng.normal(30.0, 8.0, 44)
    y = 0.6 * x + rng.normal(0.0, 11.0, 44)
    lo, hi, _ = baseline_interval(y, x, article_r=0.401, n_boot=800)
    assert hi - lo > 0.2


def test_r_stability_measures_both_distributions():
    from src.reporting.r_stability import measure

    dataset = _dataset(n_patients=18, n_epochs=2, window=750)
    s = measure(
        dataset, protocol=1, modalities=("cplx",), n_runs=2, n_splits=3,
        epochs=1, verbose=False,
    )
    assert len(s.real_r) == 2 and len(s.shuffled_r) == 2
    assert 0.0 <= s.overlap() <= 1.0
    assert np.isfinite(s.baseline_ci_lo) and np.isfinite(s.baseline_ci_hi)
    assert s.baseline_ci_lo <= s.baseline_ci_hi
