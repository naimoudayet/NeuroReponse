from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Prediction:
    patient_id: str
    valeur: int
    probabilite: float
    date: datetime
    type: str = "responder_classification"
    model_version: str | None = None
    tri_trajectory: list[float] | None = None  # per-session Therapeutic Response Index in [0, 1]
    # Regression head (the article-aligned arm): BDI-II points predicted to be
    # recovered, and the same quantity as a fraction of this patient's baseline.
    #
    # These are deliberately NOT written into `probabilite`. A predicted change of
    # 12 points is not a probability, and parking it there would let every
    # ":.1%" formatter downstream render it as "1200%" — or, worse, as a
    # plausible-looking "12%" that a clinician would read as confidence.
    delta_bdi_predit: float | None = None
    reduction_predite: float | None = None

    REGRESSION_TYPE = "bdi_regression"

    @property
    def est_regression(self) -> bool:
        return self.type == self.REGRESSION_TYPE

    @property
    def score_normalise(self) -> float:
        """The quantity comparable to the 50 % responder criterion.

        Classification returns the responder probability; regression returns the
        predicted *fraction* of BDI-II recovered. Both are on [0, 1] and both are
        thresholded at 0.5, which is what lets one comparison work for either head.
        """
        if self.est_regression:
            return float(self.reduction_predite or 0.0)
        return float(self.probabilite)

    def afficher(self) -> str:
        label = "Répondeur" if self.valeur == 1 else "Non-répondeur"
        if self.est_regression:
            return (
                f"{label} — réduction prédite {self.delta_bdi_predit:+.1f} point(s) "
                f"BDI-II ({self.score_normalise:.0%} du score initial), "
                f"patient {self.patient_id}"
            )
        return f"{label} ({self.probabilite:.1%}) — patient {self.patient_id}"

    def analyser_ecart(self, score_clinique_observe: float, seuil: float = 0.5) -> dict:
        score = self.score_normalise
        predit = 1 if score >= seuil else 0
        observe = 1 if score_clinique_observe >= seuil else 0
        return {
            "predit": predit,
            "observe": observe,
            "concordance": predit == observe,
            "ecart_probabilite": abs(score - score_clinique_observe),
        }
