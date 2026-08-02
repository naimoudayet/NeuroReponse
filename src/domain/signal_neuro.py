from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np


class SignalType(str, Enum):
    EEG = "EEG"
    FNIRS = "fNIRS"
    EMG = "EMG"
    ERP = "ERP"
    ECG = "ECG"


@dataclass
class SignalNeurophysiologique:
    type_signal: SignalType
    valeurs: np.ndarray
    timestamp: datetime
    canal: str
    sampling_rate_hz: float = 256.0
    metadata: dict = field(default_factory=dict)

    def filtrer(self, low_hz: float = 1.0, high_hz: float = 40.0) -> SignalNeurophysiologique:
        from ..preprocessing.filters import bandpass

        filtered = bandpass(self.valeurs, low_hz, high_hz, self.sampling_rate_hz)
        return SignalNeurophysiologique(
            type_signal=self.type_signal,
            valeurs=filtered,
            timestamp=self.timestamp,
            canal=self.canal,
            sampling_rate_hz=self.sampling_rate_hz,
            metadata={**self.metadata, "filtered": (low_hz, high_hz)},
        )

    def segmenter(self, window_size: int, hop: int | None = None) -> np.ndarray:
        from ..preprocessing.windowing import sliding_windows

        return sliding_windows(self.valeurs, window_size, hop or window_size)

    def extraire_features(self) -> dict[str, float]:
        """Feature dict appropriate to the modality.

        ECG is stored as an **RR tachogram** (event-sampled, ``sampling_rate_hz``
        is 0), so the spectral features that suit a uniformly-sampled trace do not
        apply — it gets time-domain HRV metrics instead. Every other type keeps the
        band-power/statistics contract.
        """
        from ..preprocessing.features import basic_features, hrv_features

        if self.type_signal is SignalType.ECG:
            return hrv_features(self.valeurs)
        return basic_features(self.valeurs, self.sampling_rate_hz)
