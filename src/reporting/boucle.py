"""Closed clinical loop: record a session, predict on the whole history, adjust, repeat.

The workflow this supports is iterative and *accumulating*:

1. The clinician records session *n* for a patient (EEG + rTMS parameters + BDI-II).
2. The model predicts using **every session so far**, not just the new one.
3. The clinician adjusts the stimulator externally — outside this application.
4. Session *n+1* is recorded, and the prediction is recomputed over ``1..n+1``.

Repeat until the response probability is satisfactory. The LSTM supports this
natively: it consumes a variable-length sequence, and ``predict_tri`` already
returns the running estimate after each timestep, so "the prediction after
session *k*" needs no retraining and no bookkeeping — it is ``tri[k-1]``.

What this module adds is the *loop's* view of that: pairing each session with the
parameters that produced it, the resulting prediction, and the change since the
previous session, so the clinician can see which adjustment moved the needle. It
holds no Streamlit imports so it stays unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..domain import RTMSParameters, SessionRTMS, SignalNeurophysiologique, SignalType

# Below this absolute change in predicted probability, a session-to-session move is
# reported as "stable" rather than as progress: the LSTM's output wobbles by a few
# points on near-identical input, and calling that an effect would mislead.
DELTA_NEGLIGEABLE = 0.02

SEUIL_REPONSE = 0.5


@dataclass
class EtapeBoucle:
    """One iteration: the session, the parameters used, and what the model then said."""

    index: int                       # 1-based session number
    id_session: str
    date: datetime
    parametres: RTMSParameters
    score_post: float | None
    tri: float | None                # prediction using sessions 1..index
    delta_tri: float | None          # change vs the previous session
    parametres_modifies: dict        # {field: (old, new)} vs the previous session

    def to_row(self) -> dict:
        return {
            "séance": self.index,
            "date": self.date.date(),
            "fréquence_hz": self.parametres.frequence_hz,
            "intensité_pct": self.parametres.intensite_pct,
            "nb_trains": self.parametres.nb_trains,
            "score_post": self.score_post,
            "P(réponse)": None if self.tri is None else round(self.tri, 3),
            "Δ": None if self.delta_tri is None else round(self.delta_tri, 3),
            "paramètres_modifiés": ", ".join(self.parametres_modifies) or "—",
        }


_CHAMPS_DOSE = (
    "frequence_hz", "intensite_pct", "duree_train_s", "nb_trains",
    "intervalle_train_s", "localisation",
)


def _diff_parametres(prev: RTMSParameters | None, cur: RTMSParameters) -> dict:
    """Which stimulation settings changed since the previous session."""
    if prev is None:
        return {}
    out = {}
    for champ in _CHAMPS_DOSE:
        a, b = getattr(prev, champ), getattr(cur, champ)
        if a != b:
            out[champ] = (a, b)
    return out


def etapes_boucle(patient, tri: list[float] | None = None) -> list[EtapeBoucle]:
    """Pair every session with its parameters and the prediction it produced.

    ``tri[k]`` is the model's estimate after session ``k+1`` — the running output
    of :meth:`ResponseLSTM.predict_tri` over the patient's full sequence.
    """
    etapes: list[EtapeBoucle] = []
    prev_params: RTMSParameters | None = None
    prev_tri: float | None = None

    for i, sess in enumerate(patient.sessions):
        val = None if tri is None or i >= len(tri) else float(tri[i])
        etapes.append(EtapeBoucle(
            index=i + 1,
            id_session=sess.id_session,
            date=sess.date,
            parametres=sess.parametres,
            score_post=sess.score_post,
            tri=val,
            delta_tri=None if (val is None or prev_tri is None) else val - prev_tri,
            parametres_modifies=_diff_parametres(prev_params, sess.parametres),
        ))
        prev_params = sess.parametres
        if val is not None:
            prev_tri = val
    return etapes


def construire_session(
    patient_id: str,
    index: int,
    parametres: RTMSParameters,
    signal: np.ndarray,
    fs: float,
    score_pre: float | None,
    score_post: float | None,
    canal: str = "Cz",
    date: datetime | None = None,
) -> SessionRTMS:
    """Assemble a new session from manually entered data.

    ``signal`` is the session's EEG window for ``canal``. It is validated here
    rather than at persist time so the caller can report a usable error before
    anything reaches the database.
    """
    x = np.asarray(signal, dtype=np.float32).ravel()
    if x.size == 0:
        raise ValueError("le signal EEG est vide")
    if not np.isfinite(x).all():
        raise ValueError("le signal EEG contient des valeurs non finies (NaN/inf)")
    if fs <= 0:
        raise ValueError("la fréquence d'échantillonnage doit être > 0")

    sess = SessionRTMS(
        id_session=f"{patient_id}-S{index:02d}",
        patient_id=patient_id,
        parametres=parametres,
        date=date or datetime.now(),
    )
    sess.enregistrer_donnees(SignalNeurophysiologique(
        type_signal=SignalType.EEG,
        valeurs=np.ascontiguousarray(x),
        timestamp=sess.date,
        canal=canal,
        sampling_rate_hz=float(fs),
    ))
    sess.score_pre = score_pre
    sess.cloturer(score_post=score_post)
    return sess


def construire_session_montage(
    patient_id: str,
    index: int,
    parametres: RTMSParameters,
    montage: dict[str, np.ndarray],
    fs: float,
    score_pre: float | None,
    score_post: float | None,
    tachogram: np.ndarray | None = None,
    ecg_canal: str = "Erbs",
    date: datetime | None = None,
) -> SessionRTMS:
    """A loop session holding a **full multi-channel recording** (TDBRAIN-style).

    Unlike :func:`construire_session` (one channel, one window), this stores the
    entire recording per channel. The epochs the model consumes are cut at
    prediction time by :func:`~src.app.inference.snapshot_input`, so one clinical
    session stays one database session — the seeded research cohort's
    "one epoch per session" layout would misrepresent a treatment visit.
    """
    if not montage:
        raise ValueError("le montage EEG est vide")
    if fs <= 0:
        raise ValueError("la fréquence d'échantillonnage doit être > 0")

    lengths = {len(v) for v in montage.values()}
    if len(lengths) != 1:
        raise ValueError(
            f"les canaux n'ont pas la même longueur ({sorted(lengths)[:3]}…)"
        )

    sess = SessionRTMS(
        id_session=f"{patient_id}-S{index:02d}",
        patient_id=patient_id,
        parametres=parametres,
        date=date or datetime.now(),
    )
    for canal, valeurs in montage.items():
        x = np.asarray(valeurs, dtype=np.float32).ravel()
        if not np.isfinite(x).all():
            raise ValueError(f"le canal {canal} contient des valeurs non finies")
        sess.enregistrer_donnees(SignalNeurophysiologique(
            type_signal=SignalType.EEG, valeurs=np.ascontiguousarray(x),
            timestamp=sess.date, canal=canal, sampling_rate_hz=float(fs),
        ))

    if tachogram is not None:
        rr = np.asarray(tachogram, dtype=np.float32).ravel()
        if rr.size and not np.isfinite(rr).all():
            raise ValueError("le tachogramme RR contient des valeurs non finies")
        # sampling_rate_hz stays 0.0: an RR series is event-sampled, not uniform.
        sess.enregistrer_donnees(SignalNeurophysiologique(
            type_signal=SignalType.ECG, valeurs=np.ascontiguousarray(rr),
            timestamp=sess.date, canal=ecg_canal, sampling_rate_hz=0.0,
        ))

    sess.score_pre = score_pre
    sess.cloturer(score_post=score_post)
    return sess


def prochain_index(patient) -> int:
    """Next session number for this patient (1-based)."""
    return len(patient.sessions) + 1


def recommandation(etapes: list[EtapeBoucle]) -> list[str]:
    """Guidance for the next iteration, derived only from what was measured.

    Deliberately conservative: it reports the direction of the last change and
    whether the target is met. It does **not** suggest specific stimulator
    settings — nothing in this project's data supports a dose-response claim (the
    TDBRAIN cohort has no per-patient dose at all), so naming an intensity would
    be invention dressed as advice.
    """
    if not etapes:
        return ["Aucune séance enregistrée."]

    msgs: list[str] = []
    avec_tri = [e for e in etapes if e.tri is not None]
    if not avec_tri:
        return ["Aucune prédiction disponible : entraîne un modèle pour cette source."]

    dernier = avec_tri[-1]
    msgs.append(
        f"Après {dernier.index} séance(s) : **P(réponse) = {dernier.tri:.0%}** "
        f"(seuil {SEUIL_REPONSE:.0%})."
    )

    if dernier.delta_tri is None:
        msgs.append("Première prédiction — enregistre une nouvelle séance pour suivre l'évolution.")
    elif dernier.delta_tri > DELTA_NEGLIGEABLE:
        modif = (", ".join(dernier.parametres_modifies) or "aucun paramètre modifié")
        msgs.append(
            f"↗ Amélioration de {dernier.delta_tri:+.0%} depuis la séance précédente "
            f"({modif}). **Conserver ces paramètres** pour la prochaine séance."
        )
    elif dernier.delta_tri < -DELTA_NEGLIGEABLE:
        modif = (", ".join(dernier.parametres_modifies) or "aucun paramètre modifié")
        msgs.append(
            f"↘ Recul de {dernier.delta_tri:+.0%} depuis la séance précédente "
            f"({modif}). **Envisager de revenir aux paramètres précédents.**"
        )
    else:
        msgs.append(
            f"→ Variation négligeable ({dernier.delta_tri:+.0%}). "
            f"Le dernier ajustement n'a pas d'effet mesurable sur la prédiction."
        )

    if dernier.tri >= SEUIL_REPONSE:
        msgs.append("✅ Objectif atteint : le modèle classe ce patient **répondeur**.")
    else:
        manque = SEUIL_REPONSE - dernier.tri
        msgs.append(
            f"⚠️ Encore {manque:.0%} sous le seuil de réponse — poursuivre le "
            f"protocole et réévaluer après la prochaine séance."
        )

    # Plateau detection: three consecutive negligible moves means the loop has
    # stopped learning anything from further identical sessions.
    recents = [e.delta_tri for e in avec_tri[-3:] if e.delta_tri is not None]
    if len(recents) >= 3 and all(abs(d) <= DELTA_NEGLIGEABLE for d in recents):
        msgs.append(
            "⚠️ **Plateau** sur les 3 dernières séances : la prédiction ne bouge "
            "plus. Un changement de protocole (fréquence ou site) est à discuter."
        )
    return msgs
