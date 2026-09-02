"""Tests for the multi-session measurement behind the clinical loop.

The loop's premise is that a prediction improves as treatment sessions
accumulate. :mod:`src.reporting.sequence_sweep` is what turns that premise into
a number, so these pin its *contract* — truncation from the front, the guard on
k, the round-trip the app reads — rather than the AUCs, which move whenever the
simulator or the training defaults do.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.reporting.sequence_sweep import (
    SequencePoint,
    read_json,
    run_point,
    write_json,
)


@pytest.fixture
def cohort():
    """A tiny separable cohort: responders carry more energy in every session."""
    rng = np.random.default_rng(0)
    n, sessions, feats = 24, 5, 8
    # Shuffled, not alternating: with one patient per group, GroupKFold hands out
    # groups in index order, so `i % 2` labels land one whole class per fold and
    # every AUC comes back NaN.
    y = np.zeros(n, dtype=np.float32)
    y[rng.permutation(n)[: n // 2]] = 1.0
    x = rng.standard_normal((n, sessions, feats)).astype(np.float32)
    x += y[:, None, None] * 2.0
    return x, y, np.arange(n)


def test_run_point_truncates_from_the_front(cohort, monkeypatch):
    """After two visits the clinic has visits 1 and 2 — never the last two."""
    x, y, groups = cohort
    vu = {}

    import src.reporting.sequence_sweep as sweep

    vrai_cv = sweep.cross_validate            # bind before patching, or it recurses

    def _fake_cv(x_in, *a, **k):
        vu["x"] = x_in
        return vrai_cv(x_in, *a, **k)

    monkeypatch.setattr(sweep, "cross_validate", _fake_cv)
    run_point(3, x, y, groups, n_splits=2, epochs=2)

    assert vu["x"].shape[1] == 3
    np.testing.assert_array_equal(vu["x"], x[:, :3, :])


@pytest.mark.parametrize("bad", [0, -1, 99])
def test_run_point_refuses_an_impossible_session_count(cohort, bad):
    x, y, groups = cohort
    with pytest.raises(ValueError, match="n_sessions"):
        run_point(bad, x, y, groups, n_splits=2, epochs=2)


def test_run_point_reports_the_cohort_it_measured(cohort):
    x, y, groups = cohort
    point = run_point(2, x, y, groups, n_splits=2, epochs=2)
    assert point.n_sessions == 2
    assert point.n_patients == x.shape[0]
    assert point.n_features == x.shape[-1]
    assert point.base_rate == pytest.approx(0.5)
    assert 0.0 <= point.auc_mean <= 1.0


def test_more_sessions_do_not_lose_information(cohort):
    """The whole premise: a longer course must not predict *worse* than one visit.

    Deliberately a loose bound — the point is directional, and pinning an exact
    AUC here would break every time the training defaults are touched.
    """
    x, y, groups = cohort
    une = run_point(1, x, y, groups, n_splits=2, epochs=12).auc_mean
    toutes = run_point(x.shape[1], x, y, groups, n_splits=2, epochs=12).auc_mean
    assert toutes >= une - 0.15


def test_json_round_trips_for_the_app(tmp_path):
    """The comparison page reads this file; a schema drift would blank the panel."""
    path = tmp_path / "sequence_sweep.json"
    points = [
        SequencePoint(
            n_sessions=k, auc_mean=0.9, auc_std=0.01, accuracy_mean=0.88,
            accuracy_std=0.02, f1_mean=0.87, base_rate=0.5,
            n_patients=100, n_features=8,
        )
        for k in (1, 2)
    ]
    write_json(points, path)

    lus = read_json(path)
    assert [p["n_sessions"] for p in lus] == [1, 2]
    # Exactly the keys the page indexes into.
    for cle in ("n_sessions", "auc_mean", "auc_std", "accuracy_mean", "base_rate"):
        assert cle in lus[0]


def test_read_json_is_empty_when_never_run(tmp_path):
    """The page must show its 'run this' hint, not crash, on a fresh clone."""
    assert read_json(tmp_path / "absent.json") == []
