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
    # Réel, pas entier : TDBRAIN publie des âges décimaux (49.66) et l'âge est le
    # prédicteur le plus fort du projet. L'arrondir à la sauvegarde ferait prédire
    # le modèle sur une valeur qu'il n'a jamais vue à l'entraînement. L'affichage
    # arrondit ; le stockage non.
    age: float
    diagnostic: str
    # Encodé 0/1 comme dans les tables sources (TDBRAIN `gender`, simulateur
    # apparié), et non "H"/"F" : c'est la valeur telle quelle qui alimente le bloc
    # clinique du modèle, donc la stocker autrement obligerait à un recodage qui
    # pourrait diverger de l'entraînement. `None` = non renseigné, jamais 0.
    sexe: int | None = None
    historique_clinique: list[DossierClinique] = field(default_factory=list)
    sessions: list[SessionRTMS] = field(default_factory=list)

    def ajouter_dossier(self, dossier: DossierClinique) -> None:
        self.historique_clinique.append(dossier)

    def consulter_historique(self) -> list[DossierClinique]:
        return list(self.historique_clinique)

    def ajouter_session(self, session: SessionRTMS) -> None:
        self.sessions.append(session)
