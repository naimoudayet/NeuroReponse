from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session_rtms import SessionRTMS


@dataclass
class DossierClinique:
    date: datetime
    note: str
    score_depression: float | None = None


@dataclass
class Patient:
    id: str
    nom: str
    age: int
    diagnostic: str
    historique_clinique: list[DossierClinique] = field(default_factory=list)
    sessions: list[SessionRTMS] = field(default_factory=list)

    def ajouter_dossier(self, dossier: DossierClinique) -> None:
        self.historique_clinique.append(dossier)

    def consulter_historique(self) -> list[DossierClinique]:
        return list(self.historique_clinique)

    def ajouter_session(self, session: SessionRTMS) -> None:
        self.sessions.append(session)
