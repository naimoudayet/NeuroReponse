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

    def afficher(self) -> str:
        label = "Répondeur" if self.valeur == 1 else "Non-répondeur"
        return f"{label} ({self.probabilite:.1%}) — patient {self.patient_id}"

    def analyser_ecart(self, score_clinique_observe: float, seuil: float = 0.5) -> dict:
        predit = 1 if self.probabilite >= seuil else 0
        observe = 1 if score_clinique_observe >= seuil else 0
        return {
            "predit": predit,
            "observe": observe,
            "concordance": predit == observe,
            "ecart_probabilite": abs(self.probabilite - score_clinique_observe),
        }
