"""Tests for the closed clinical loop (record -> predict on all sessions -> adjust)."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.domain import Patient, RTMSParameters, SignalType
from src.reporting.boucle import (
    DELTA_NEGLIGEABLE,
    construire_session,
    etapes_boucle,
    prochain_index,
    recommandation,
)


def _params(intensite=110.0, freq=10.0, trains=75):
    return RTMSParameters(
        frequence_hz=freq, intensite_pct=intensite, duree_train_s=4.0,
        nb_trains=trains, intervalle_train_s=26.0, localisation="DLPFC_gauche",
    )


def _patient(param_list, scores=None, pid="P1"):
    p = Patient(id=pid, nom="Test", age=40, diagnostic="MDD")
    for i, prm in enumerate(param_list):
        sess = construire_session(
            patient_id=pid, index=i + 1, parametres=prm,
            signal=np.random.default_rng(i).standard_normal(128),
            fs=256.0, score_pre=30.0,
            score_post=None if scores is None else scores[i],
            date=datetime(2026, 1, 1) + timedelta(days=i),
        )
        p.ajouter_session(sess)
    return p


def test_construire_session_builds_a_persistable_session():
    sess = construire_session(
        patient_id="P9", index=3, parametres=_params(),
        signal=np.ones(128), fs=256.0, score_pre=30.0, score_post=22.0,
    )
    assert sess.id_session == "P9-S03"
    assert sess.patient_id == "P9"
    assert sess.statut == "terminee"
    assert sess.score_post == 22.0
    assert len(sess.signaux) == 1
    assert sess.signaux[0].type_signal is SignalType.EEG
    assert sess.signaux[0].valeurs.shape == (128,)
    assert sess.signaux[0].sampling_rate_hz == 256.0


@pytest.mark.parametrize("bad, match", [
    (np.array([]), "vide"),
    (np.array([1.0, np.nan, 3.0]), "non finies"),
])
def test_construire_session_rejects_bad_signals(bad, match):
    with pytest.raises(ValueError, match=match):
        construire_session("P1", 1, _params(), bad, 256.0, 30.0, 25.0)


def test_construire_session_rejects_bad_sampling_rate():
    with pytest.raises(ValueError, match="échantillonnage"):
        construire_session("P1", 1, _params(), np.ones(64), 0.0, 30.0, 25.0)


def test_prochain_index_continues_the_sequence():
    p = _patient([_params(), _params()])
    assert prochain_index(p) == 3


def test_prochain_index_reads_the_ids_not_the_count():
    """A gap in the numbering must not make the loop re-issue an existing id.

    Counting sessions instead of reading their numbers handed out an id that a
    stored session already owned, and `_upsert_session` would then overwrite that
    session rather than appending a new one — a recorded visit silently lost.
    """
    p = _patient([_params(), _params(), _params()])
    del p.sessions[1]                                   # S01, S03 remain
    assert prochain_index(p) == 4


def test_etapes_pair_each_session_with_its_running_prediction():
    """tri[k] is the prediction using sessions 1..k+1 — the accumulating loop."""
    p = _patient([_params(100.0), _params(110.0), _params(120.0)],
                 scores=[29.0, 25.0, 18.0])
    etapes = etapes_boucle(p, tri=[0.30, 0.45, 0.62])

    assert [e.index for e in etapes] == [1, 2, 3]
    assert [e.tri for e in etapes] == [0.30, 0.45, 0.62]
    assert etapes[0].delta_tri is None                 # nothing to compare against
    assert etapes[1].delta_tri == pytest.approx(0.15)
    assert etapes[2].delta_tri == pytest.approx(0.17)


def test_etapes_report_which_parameters_the_clinician_changed():
    p = _patient([_params(100.0, trains=75), _params(120.0, trains=75),
                  _params(120.0, trains=90)])
    etapes = etapes_boucle(p, tri=[0.3, 0.5, 0.6])

    assert etapes[0].parametres_modifies == {}
    assert "intensite_pct" in etapes[1].parametres_modifies
    assert etapes[1].parametres_modifies["intensite_pct"] == (100.0, 120.0)
    assert list(etapes[2].parametres_modifies) == ["nb_trains"]


def test_etapes_tolerate_a_missing_prediction():
    p = _patient([_params(), _params()])
    etapes = etapes_boucle(p, tri=None)
    assert all(e.tri is None and e.delta_tri is None for e in etapes)
    assert etapes[0].to_row()["P(réponse)"] is None


def test_recommendation_keeps_settings_when_improving():
    p = _patient([_params(100.0), _params(120.0)])
    msgs = recommandation(etapes_boucle(p, tri=[0.30, 0.55]))
    joined = " ".join(msgs)
    assert "Amélioration" in joined
    assert "Conserver ces paramètres" in joined
    assert "intensite_pct" in joined
    assert "Objectif atteint" in joined            # 0.55 >= 0.5


def test_recommendation_suggests_reverting_when_worse():
    p = _patient([_params(100.0), _params(140.0)])
    msgs = recommandation(etapes_boucle(p, tri=[0.48, 0.20]))
    joined = " ".join(msgs)
    assert "Recul" in joined
    assert "revenir aux paramètres précédents" in joined
    assert "sous le seuil" in joined


def test_recommendation_calls_small_moves_negligible():
    p = _patient([_params(), _params()])
    msgs = recommandation(etapes_boucle(p, tri=[0.40, 0.40 + DELTA_NEGLIGEABLE / 2]))
    assert any("négligeable" in m for m in msgs)


def test_recommendation_detects_a_plateau():
    p = _patient([_params()] * 5)
    msgs = recommandation(etapes_boucle(p, tri=[0.30, 0.305, 0.31, 0.305, 0.30]))
    assert any("Plateau" in m for m in msgs)


def test_recommendation_without_a_model():
    p = _patient([_params()])
    msgs = recommandation(etapes_boucle(p, tri=None))
    assert any("Aucune prédiction" in m for m in msgs)


def test_recommendation_on_empty_history():
    assert recommandation([]) == ["Aucune séance enregistrée."]


def test_loop_grows_the_sequence_the_model_consumes():
    """The whole point: adding a session lengthens the model's input by one step."""
    import torch

    from src.models.lstm import LSTMConfig, ResponseLSTM

    model = ResponseLSTM(LSTMConfig(input_size=8))
    lengths = []
    for n in (1, 2, 3, 4):
        x = torch.randn(1, n, 8)
        tri = model.predict_tri(x)
        lengths.append(tri.shape[1])
        # The final TRI value must equal the one-shot probability for that history.
        assert float(tri[0, -1]) == pytest.approx(float(model.predict_proba(x)), abs=1e-6)
    assert lengths == [1, 2, 3, 4]


def test_recommandation_never_calls_a_predicted_reduction_a_probability():
    """The regression head's value is an improvement, not a confidence.

    Both heads land on [0, 1] against the 50 % criterion, which is what lets one
    function serve both — but printing "P(réponse) = 29 %" for a model that is
    predicting a 29 % *improvement* tells the clinician something different, and
    wrong.
    """
    p = _patient([_params(), _params()], scores=[28.0, 26.0])
    etapes = etapes_boucle(p, tri=[0.20, 0.29])

    par_defaut = recommandation(etapes)
    assert any("P(réponse) = 29%" in m for m in par_defaut)

    regression = recommandation(etapes, libelle="Réduction BDI-II prédite")
    assert any("Réduction BDI-II prédite = 29%" in m for m in regression)
    assert not any("P(réponse)" in m for m in regression)
