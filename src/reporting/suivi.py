"""Longitudinal follow-up: synthesise a patient's whole course, not one session.

The Predictions page answers "will this patient respond?" from the final LSTM
output. This module answers the clinician's other question — "how is this patient
*going*?" — by reading **every** session at once: the clinical score trajectory,
the per-session model trajectory (TRI), and whether the two agree.

The central subtlety is that "session" means different things per cohort, and the
synthesis must not pretend otherwise:

* **Simulated cohort** — 10 genuine rTMS sessions with a per-session ``score_post``
  that actually moves. A trend over these is a real clinical trajectory.
* **TDBRAIN** — one baseline resting recording chopped into *epochs*. ``score_pre``
  and ``score_post`` are identical on every epoch because there is only one
  treatment course. A "trend" over them would be pure fiction.

:func:`analyser_suivi` therefore detects whether the scores actually vary
(:attr:`SyntheseSuivi.trajectoire_clinique_disponible`) and downgrades its own
claims when they do not: no clinical trend, and the TRI spread is reported as
*model stability across epochs* rather than as progress over time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# A session-to-session change smaller than this (in BDI-II points) is treated as
# flat: the simulated generator jitters scores by well under a point even for
# non-responders, and calling that "amélioration" would be noise-chasing.
PENTE_PLATE = 0.05

# Standard BDI-II response criterion, shared with the loader's responder label.
SEUIL_REPONSE = 0.5


@dataclass
class SyntheseSuivi:
    """Everything the follow-up view shows, computed from *all* sessions."""

    n_sessions: int
    unite: str                                   # "séance" or "époque"
    trajectoire_clinique_disponible: bool

    score_initial: float | None = None
    score_final: float | None = None
    reduction_pct: float | None = None
    repondeur_observe: bool | None = None
    pente_par_session: float | None = None       # BDI-II points per session
    tendance: str | None = None                  # amélioration / stable / aggravation

    tri_final: float | None = None
    tri_moyen: float | None = None
    tri_ecart_type: float | None = None
    tri_pente: float | None = None
    tri_stable: bool | None = None

    coherence: str | None = None                 # model vs clinical agreement
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["messages"] = list(self.messages)
        return d


def _pente(y: np.ndarray) -> float:
    """Least-squares slope per index step; 0.0 when there is nothing to fit."""
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=np.float64)
    return float(np.polyfit(x, y.astype(np.float64), 1)[0])


def trajectoire_scores(patient) -> list[float | None]:
    """Per-session ``score_post``, in session order — the clinical trajectory."""
    return [s.score_post for s in patient.sessions]


def analyser_suivi(
    patient,
    tri_trajectory: list[float] | None = None,
    unite: str = "séance",
) -> SyntheseSuivi:
    """Summarise a patient's whole course from every session.

    ``tri_trajectory`` is the model's per-session response probability (the LSTM's
    Therapeutic Response Index). Pass ``None`` to build a clinical-only synthesis.
    """
    sessions = list(patient.sessions)
    scores = [s for s in trajectoire_scores(patient) if s is not None]

    # A trajectory exists only if the post-treatment score actually moves. TDBRAIN
    # repeats one course across epochs, so every value is identical.
    varie = len(scores) >= 2 and float(np.ptp(np.asarray(scores, dtype=float))) > 1e-9
    syn = SyntheseSuivi(
        n_sessions=len(sessions),
        unite=unite,
        trajectoire_clinique_disponible=varie,
    )

    baseline = sessions[0].score_pre if sessions else None
    if baseline is not None and scores:
        syn.score_initial = float(baseline)
        syn.score_final = float(scores[-1])
        if baseline > 0:
            syn.reduction_pct = float((baseline - scores[-1]) / baseline)
            syn.repondeur_observe = bool(syn.reduction_pct >= SEUIL_REPONSE)

    if varie:
        arr = np.asarray(scores, dtype=float)
        syn.pente_par_session = _pente(arr)
        if syn.pente_par_session < -PENTE_PLATE:
            syn.tendance = "amélioration"
        elif syn.pente_par_session > PENTE_PLATE:
            syn.tendance = "aggravation"
        else:
            syn.tendance = "stable"

    if tri_trajectory:
        tri = np.asarray(tri_trajectory, dtype=float)
        syn.tri_final = float(tri[-1])
        syn.tri_moyen = float(tri.mean())
        syn.tri_ecart_type = float(tri.std())
        syn.tri_pente = _pente(tri)
        syn.tri_stable = bool(syn.tri_ecart_type < 0.10)

    syn.coherence = _coherence(syn)
    syn.messages = _messages(syn)
    return syn


def _coherence(syn: SyntheseSuivi) -> str | None:
    """Do the model's verdict and the observed outcome agree?

    Compared on the *outcome*, not on trend direction: the TRI is a response
    probability, so its slope has no clinical unit to match against BDI points.
    """
    if syn.tri_final is None or syn.repondeur_observe is None:
        return None
    predit = syn.tri_final >= SEUIL_REPONSE
    return "concordant" if predit == syn.repondeur_observe else "discordant"


def _messages(syn: SyntheseSuivi) -> list[str]:
    """Clinician-facing feedback, each line traceable to a computed field."""
    out: list[str] = []

    if not syn.trajectoire_clinique_disponible:
        out.append(
            f"Aucune trajectoire clinique : les {syn.n_sessions} {syn.unite}s portent "
            f"des scores identiques. Il s'agit d'un **enregistrement unique découpé "
            f"en {syn.unite}s**, pas d'un suivi longitudinal — aucune évolution ne "
            f"peut être calculée."
        )
    elif syn.tendance == "amélioration":
        out.append(
            f"Tendance à l'**amélioration** : {abs(syn.pente_par_session):.2f} point "
            f"BDI-II gagné par {syn.unite} en moyenne sur {syn.n_sessions} {syn.unite}s."
        )
    elif syn.tendance == "aggravation":
        out.append(
            f"⚠️ Tendance à l'**aggravation** : +{syn.pente_par_session:.2f} point "
            f"BDI-II par {syn.unite}. Réévaluer le protocole."
        )
    else:
        out.append(
            f"Évolution **stable** sur {syn.n_sessions} {syn.unite}s "
            f"(pente {syn.pente_par_session:+.2f} point/{syn.unite})."
        )

    if syn.reduction_pct is not None:
        verdict = "répondeur" if syn.repondeur_observe else "non-répondeur"
        # One decimal when the value rounds to zero but is not zero, so a slight
        # worsening does not print as the meaningless "-0%".
        pct = syn.reduction_pct
        pct_txt = f"{pct:.1%}" if 0 < abs(pct) < 0.005 else f"{pct:.0%}"
        out.append(
            f"Réduction totale {pct_txt} "
            f"({syn.score_initial:.1f} → {syn.score_final:.1f}) → **{verdict}** "
            f"(seuil {SEUIL_REPONSE:.0%})."
        )

    if syn.tri_final is not None:
        if syn.trajectoire_clinique_disponible:
            spread = (
                f"stable (σ = {syn.tri_ecart_type:.02f})" if syn.tri_stable
                else f"instable (σ = {syn.tri_ecart_type:.02f})"
            )
            out.append(
                f"Modèle : TRI final {syn.tri_final:.0%}, moyenne "
                f"{syn.tri_moyen:.0%}, {spread} sur les {syn.unite}s."
            )
        else:
            # No time axis: the spread measures agreement between windows of the
            # same recording, which is a confidence signal, not a progress signal.
            out.append(
                f"Modèle : TRI final {syn.tri_final:.0%}. L'écart-type "
                f"({syn.tri_ecart_type:.02f}) mesure la **cohérence entre "
                f"{syn.unite}s du même enregistrement**, pas une évolution."
            )
        if syn.tri_stable is False:
            out.append(
                f"⚠️ Prédiction peu stable d'une {syn.unite} à l'autre : "
                f"interpréter le TRI final avec prudence."
            )

    if syn.coherence == "concordant":
        out.append("✅ Prédiction **concordante** avec l'issue observée.")
    elif syn.coherence == "discordant":
        out.append("❌ Prédiction **discordante** avec l'issue observée.")

    return out
