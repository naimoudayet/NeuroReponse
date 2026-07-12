"""Simulated EEG dataset for the rTMS + LSTM prototype.

Each "patient" has 10 rTMS sessions. Each session yields 128 EEG samples
(0.5 s @ 256 Hz) built from a mixture of band-limited oscillations plus noise.

Responders and non-responders differ in how their alpha-band power evolves
across sessions:

    responder:     alpha power grows session-over-session (treatment works)
    non-responder: alpha power stays flat / drifts randomly

The shift is intentionally small so the LSTM has to integrate evidence
across sessions to detect it — that's the point of the model.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_FS = 256.0
DEFAULT_WINDOW = 128
DEFAULT_N_PATIENTS = 100
DEFAULT_N_SESSIONS = 10


@dataclass(frozen=True)
class SimConfig:
    n_patients: int = DEFAULT_N_PATIENTS
    n_sessions: int = DEFAULT_N_SESSIONS
    window: int = DEFAULT_WINDOW
    fs: float = DEFAULT_FS
    responder_rate: float = 0.5
    noise_std: float = 0.6
    responder_alpha_gain: float = 0.05  # per-session alpha growth for responders
    # --- multimodal (ERP + ECG) — see RES0_AR1 / NPDT: cognitive + autonomic channels ---
    multimodal: bool = True
    n_rr: int = 64                        # RR intervals per session (ECG tachogram length)
    responder_erp_gain: float = 0.10      # per-session N100/P300 growth for responders
    responder_hrv_gain: float = 0.08      # per-session HRV growth for responders
    seed: int = 42


@dataclass
class SimulatedDataset:
    signals: np.ndarray          # (n_patients, n_sessions, window)  float32  — EEG
    labels: np.ndarray           # (n_patients,)  int8 in {0, 1}
    metadata: pd.DataFrame       # one row per (patient, session)
    config: SimConfig
    erp: np.ndarray | None = None    # (n_patients, n_sessions, window)  float32  — evoked potential
    ecg: np.ndarray | None = None    # (n_patients, n_sessions, n_rr)   float32  — RR intervals (s)


def _osc(t: np.ndarray, freq: float, amp: float, phase: float) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * t + phase)


def _gauss(t: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def _make_erp(t: np.ndarray, n100_amp: float, p300_amp: float, rng: np.random.Generator) -> np.ndarray:
    """Averaged evoked potential: a negative N100 (~100 ms) + a positive P300 (~300 ms)."""
    wave = -n100_amp * _gauss(t, 0.10, 0.02) + p300_amp * _gauss(t, 0.30, 0.05)
    wave = wave + 0.05 * rng.standard_normal(t.shape[0])
    return wave.astype(np.float32)


def _make_rr(n_rr: int, mean_rr: float, sdnn_s: float, rng: np.random.Generator) -> np.ndarray:
    """Synthetic RR-interval tachogram (seconds). White beat-to-beat variability → SDNN≈RMSSD/√2."""
    rr = mean_rr + sdnn_s * rng.standard_normal(n_rr)
    return np.clip(rr, 0.3, 2.0).astype(np.float32)  # keep physiological (30–200 bpm)


def simulate(cfg: SimConfig | None = None) -> SimulatedDataset:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    # ERP/ECG draw from an INDEPENDENT stream so the EEG signal stays byte-identical
    # whether or not multimodal generation is enabled.
    rng_m = np.random.default_rng(cfg.seed + 10_007) if cfg.multimodal else None

    t = np.arange(cfg.window) / cfg.fs

    n_responders = int(round(cfg.n_patients * cfg.responder_rate))
    labels = np.array([1] * n_responders + [0] * (cfg.n_patients - n_responders), dtype=np.int8)
    rng.shuffle(labels)

    signals = np.empty((cfg.n_patients, cfg.n_sessions, cfg.window), dtype=np.float32)
    erp = np.empty((cfg.n_patients, cfg.n_sessions, cfg.window), dtype=np.float32) if cfg.multimodal else None
    ecg = np.empty((cfg.n_patients, cfg.n_sessions, cfg.n_rr), dtype=np.float32) if cfg.multimodal else None
    meta_rows: list[dict] = []

    for p in range(cfg.n_patients):
        is_responder = bool(labels[p])

        # Per-patient baseline amplitudes per band (μV-ish, arbitrary units).
        baseline = {
            "theta": rng.uniform(0.5, 1.2),
            "alpha": rng.uniform(0.8, 1.6),
            "beta":  rng.uniform(0.3, 0.8),
        }
        # Per-patient clinical score (e.g. HDRS), starts moderate-severe.
        score = rng.uniform(20, 30)

        # Per-patient ERP/ECG baselines (independent RNG stream).
        if rng_m is not None:
            erp_n100_base = rng_m.uniform(0.8, 1.4)   # N100 deflection magnitude
            erp_p300_base = rng_m.uniform(0.6, 1.2)   # P300 amplitude
            hrv_base = rng_m.uniform(0.020, 0.045)    # SDNN-ish (s)
            mean_rr = rng_m.uniform(0.75, 0.90)       # ~67–80 bpm

        for s in range(cfg.n_sessions):
            # Alpha modulation across sessions.
            if is_responder:
                alpha_gain = 1.0 + cfg.responder_alpha_gain * s
            else:
                alpha_gain = 1.0 + 0.01 * rng.standard_normal()

            theta = _osc(t, freq=6.0, amp=baseline["theta"], phase=rng.uniform(0, 2 * np.pi))
            alpha = _osc(t, freq=10.0, amp=baseline["alpha"] * alpha_gain,
                         phase=rng.uniform(0, 2 * np.pi))
            beta  = _osc(t, freq=20.0, amp=baseline["beta"], phase=rng.uniform(0, 2 * np.pi))
            noise = cfg.noise_std * rng.standard_normal(cfg.window)

            signals[p, s] = (theta + alpha + beta + noise).astype(np.float32)

            # Simulate clinical score evolution.
            if is_responder:
                score = max(0.0, score - rng.uniform(0.5, 1.5))
            else:
                score = max(0.0, score + rng.normal(0.0, 0.7))

            row = {
                "patient_id": f"P{p:03d}",
                "session_idx": s,
                "label": int(is_responder),
                "alpha_gain": float(alpha_gain),
                "score_clinique": float(score),
                "frequence_hz": 10.0,
                "intensite_pct": 110.0,
                "localisation": "DLPFC_gauche",
            }

            # Cognitive (ERP) + autonomic (ECG) modalities grow with recovery for responders.
            if rng_m is not None:
                if is_responder:
                    erp_scale = 1.0 + cfg.responder_erp_gain * s
                    hrv_scale = 1.0 + cfg.responder_hrv_gain * s
                else:
                    erp_scale = 1.0 + 0.02 * rng_m.standard_normal()
                    hrv_scale = 1.0 + 0.02 * rng_m.standard_normal()

                n100 = erp_n100_base * erp_scale
                p300 = erp_p300_base * erp_scale
                sdnn = hrv_base * hrv_scale
                erp[p, s] = _make_erp(t, n100, p300, rng_m)
                ecg[p, s] = _make_rr(cfg.n_rr, mean_rr, sdnn, rng_m)
                row["erp_n100_uv"] = float(n100)
                row["hrv_sdnn_ms"] = float(sdnn * 1000.0)

            meta_rows.append(row)

    metadata = pd.DataFrame(meta_rows)
    return SimulatedDataset(
        signals=signals, labels=labels, metadata=metadata, config=cfg, erp=erp, ecg=ecg
    )


def save(dataset: SimulatedDataset, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "eeg_simulated.npz"
    csv_path = out_dir / "metadata.csv"

    extra = {}
    if dataset.erp is not None:
        extra["erp"] = dataset.erp
    if dataset.ecg is not None:
        extra["ecg"] = dataset.ecg

    np.savez_compressed(
        npz_path,
        signals=dataset.signals,
        labels=dataset.labels,
        fs=np.float32(dataset.config.fs),
        window=np.int32(dataset.config.window),
        **extra,
    )
    dataset.metadata.to_csv(csv_path, index=False)
    return npz_path, csv_path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate simulated rTMS+EEG dataset.")
    ap.add_argument("--out", type=Path, default=Path("data/simulated"))
    ap.add_argument("--patients", type=int, default=DEFAULT_N_PATIENTS)
    ap.add_argument("--sessions", type=int, default=DEFAULT_N_SESSIONS)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = SimConfig(
        n_patients=args.patients,
        n_sessions=args.sessions,
        window=args.window,
        seed=args.seed,
    )
    ds = simulate(cfg)
    npz_path, csv_path = save(ds, args.out)
    n_resp = int(ds.labels.sum())
    print(
        f"Generated {cfg.n_patients} patients x {cfg.n_sessions} sessions "
        f"x {cfg.window} samples ({n_resp} responders, {cfg.n_patients - n_resp} non-responders)"
    )
    print(f"  signals -> {npz_path}")
    print(f"  metadata -> {csv_path}")


if __name__ == "__main__":
    main()
