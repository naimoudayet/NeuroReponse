from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .signal_neuro import SignalNeurophysiologique


@dataclass
class Preprocessing:
    techniques: list[str] = field(default_factory=lambda: ["bandpass", "normalize", "window"])
    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 40.0
    window_size: int = 128

    def nettoyer(self, signal: SignalNeurophysiologique) -> SignalNeurophysiologique:
        return signal.filtrer(self.bandpass_low_hz, self.bandpass_high_hz)

    def normaliser(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / np.where(std == 0, 1.0, std)

    def fenetrer(self, signal: SignalNeurophysiologique) -> np.ndarray:
        return signal.segmenter(self.window_size)

    def pipeline(self, signal: SignalNeurophysiologique) -> np.ndarray:
        cleaned = self.nettoyer(signal)
        windows = self.fenetrer(cleaned)
        return self.normaliser(windows)

    def pipeline_dataset(
        self,
        signals: np.ndarray,
        fs: float,
        mode: str = "features",
    ):
        from ..preprocessing.pipeline import PipelineConfig, preprocess

        cfg = PipelineConfig(
            fs=fs,
            bandpass_low_hz=self.bandpass_low_hz,
            bandpass_high_hz=self.bandpass_high_hz,
            mode=mode,  # type: ignore[arg-type]
        )
        return preprocess(signals, cfg)
