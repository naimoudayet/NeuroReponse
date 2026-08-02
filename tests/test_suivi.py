"""Tests for the longitudinal follow-up synthesis.

The guarantee that matters here is honesty about the time axis: TDBRAIN's
"sessions" are epochs of one recording, so the synthesis must refuse to report a
clinical trend for them while still reporting one for the genuinely longitudinal
simulated cohort.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.domain import Patient, RTMSParameters, SessionRTMS
from src.reporting.suivi import SEUIL_REPONSE, analyser_suivi, trajectoire_scores

PARAMS = RTMSParameters(
    frequence_hz=10.0, intensite_pct=110.0, duree_train_s=4.0,
    nb_trains=75, intervalle_train_s=26.0, localisation="DLPFC_gauche",
)


def _patient(scores_post, score_pre=30.0, pid="P1"):
    """Build a patient whose sessions carry the given per-session post scores."""
    p = Patient(id=pid, nom="Test", age=40, diagnostic="MDD")
    for i, post in enumerate(scores_post):
        s = SessionRTMS(
            id_session=f"{pid}-S{i:02d}", patient_id=pid, parametres=PARAMS,
            date=datetime(2026, 1, 1) + timedelta(days=i),
        )
        s.score_pre = score_pre
        s.cloturer(score_post=post)
        p.ajouter_session(s)
    return p


def test_improving_course_is_detected():
    p = _patient([30.0, 26.0, 22.0, 18.0, 12.0])       # 30 -> 12 = 60% reduction
    syn = analyser_suivi(p, tri_trajectory=[0.5, 0.6, 0.7, 0.8, 0.85])

    assert syn.trajectoire_clinique_disponible is True
    assert syn.tendance == "amélioration"
    assert syn.pente_par_session < 0
    assert syn.reduction_pct == pytest.approx(0.6)
    assert syn.repondeur_observe is True
    assert syn.coherence == "concordant"
    assert any("amélioration" in m for m in syn.messages)


def test_worsening_course_is_flagged():
    p = _patient([30.0, 31.0, 32.5, 34.0])
    syn = analyser_suivi(p, tri_trajectory=[0.4, 0.35, 0.3, 0.2])

    assert syn.tendance == "aggravation"
    assert syn.pente_par_session > 0
    assert syn.repondeur_observe is False
    assert syn.coherence == "concordant"          # model also says non-responder
    assert any("aggravation" in m for m in syn.messages)


def test_flat_jitter_is_reported_as_stable_not_a_trend():
    """Sub-0.05-point wobble is noise; calling it a trend would mislead."""
    p = _patient([29.00, 29.02, 28.99, 29.01, 29.00])
    syn = analyser_suivi(p)

    assert syn.trajectoire_clinique_disponible is True
    assert syn.tendance == "stable"
    assert syn.repondeur_observe is False


def test_identical_scores_report_no_clinical_trajectory():
    """TDBRAIN: 8 epochs of one recording all carry the same pre/post scores."""
    p = _patient([8.0] * 8, score_pre=20.0)
    syn = analyser_suivi(p, tri_trajectory=[0.6] * 8, unite="époque")

    assert syn.trajectoire_clinique_disponible is False
    assert syn.tendance is None
    assert syn.pente_par_session is None
    # The overall outcome is still known — one course, measured once.
    assert syn.reduction_pct == pytest.approx(0.6)
    assert syn.repondeur_observe is True
    assert any("pas d'un suivi longitudinal" in m for m in syn.messages)
    # And it must not claim progress from the epoch axis.
    assert not any("amélioration" in m or "aggravation" in m for m in syn.messages)


def test_epoch_spread_is_described_as_coherence_not_progress():
    p = _patient([8.0] * 8, score_pre=20.0)
    syn = analyser_suivi(p, tri_trajectory=[0.61, 0.59, 0.62, 0.60, 0.58, 0.61, 0.60, 0.59],
                         unite="époque")
    joined = " ".join(syn.messages)
    assert "cohérence entre" in joined
    assert "évolution" in joined            # explicitly denies it is one


def test_unstable_prediction_is_warned_about():
    p = _patient([30.0, 25.0, 20.0, 14.0])
    syn = analyser_suivi(p, tri_trajectory=[0.10, 0.90, 0.15, 0.85])

    assert syn.tri_stable is False
    assert any("peu stable" in m for m in syn.messages)


def test_discordance_between_model_and_outcome():
    p = _patient([30.0, 29.0, 28.5, 28.0])          # only 6.7% reduction
    syn = analyser_suivi(p, tri_trajectory=[0.8, 0.85, 0.9, 0.92])

    assert syn.repondeur_observe is False
    assert syn.tri_final >= SEUIL_REPONSE
    assert syn.coherence == "discordant"
    assert any("discordante" in m for m in syn.messages)


def test_works_without_a_model_trajectory():
    p = _patient([30.0, 24.0, 18.0])
    syn = analyser_suivi(p)

    assert syn.tri_final is None
    assert syn.coherence is None
    assert syn.tendance == "amélioration"
    assert syn.messages


def test_trajectoire_scores_preserves_session_order():
    p = _patient([30.0, 20.0, 10.0])
    assert trajectoire_scores(p) == [30.0, 20.0, 10.0]


def test_single_session_has_no_trend():
    p = _patient([15.0], score_pre=30.0)
    syn = analyser_suivi(p, tri_trajectory=[0.7])

    assert syn.trajectoire_clinique_disponible is False
    assert syn.reduction_pct == pytest.approx(0.5)
    assert syn.repondeur_observe is True
