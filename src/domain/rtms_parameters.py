from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RTMSParameters:
    frequence_hz: float
    intensite_pct: float
    duree_train_s: float
    nb_trains: int
    intervalle_train_s: float
    localisation: str
    protocole: str = "standard"

    def total_pulses(self) -> int:
        return int(self.frequence_hz * self.duree_train_s * self.nb_trains)
