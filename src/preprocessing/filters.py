from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass(x: np.ndarray, low_hz: float, high_hz: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=-1)
