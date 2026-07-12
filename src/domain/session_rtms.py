from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .rtms_parameters import RTMSParameters
from .signal_neuro import SignalNeurophysiologique


@dataclass
class SessionRTMS:
    id_session: str
    patient_id: str
    parametres: RTMSParameters
    date: datetime = field(default_factory=datetime.now)
    signaux: list[SignalNeurophysiologique] = field(default_factory=list)
    score_pre: float | None = None
    score_post: float | None = None
    statut: str = "planifiee"

    def demarrer(self) -> None:
        self.statut = "en_cours"
        self.date = datetime.now()

    def enregistrer_donnees(self, signal: SignalNeurophysiologique) -> None:
        self.signaux.append(signal)

    def cloturer(self, score_post: float | None = None) -> None:
        self.statut = "terminee"
        if score_post is not None:
            self.score_post = score_post

    def generer_rapport(self) -> dict:
        return {
            "id_session": self.id_session,
            "patient_id": self.patient_id,
            "date": self.date.isoformat(),
            "statut": self.statut,
            "parametres": {
                "frequence_hz": self.parametres.frequence_hz,
                "intensite_pct": self.parametres.intensite_pct,
                "localisation": self.parametres.localisation,
                "total_pulses": self.parametres.total_pulses(),
            },
            "nb_signaux": len(self.signaux),
            "score_pre": self.score_pre,
            "score_post": self.score_post,
        }
