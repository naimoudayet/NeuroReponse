"""TDBRAIN-matched simulator: same shape, same statistics, controllable signal.

The original :mod:`src.data.simulator` produces a 1-channel, 100-patient cohort of
its own design. That is fine on its own terms but cannot be compared with TDBRAIN:
different montage, different cohort size, different feature count. This module
generates a cohort that is **structurally identical** to the real one — 132
patients, 8 epochs, the 26-channel montage, an ECG tachogram, and matching
clinical/demographic marginals — and returns the very same
:class:`~src.data.loader.LoadedDataset` contract that :func:`load_tdbrain`
returns, so features, training, seeding and the app all work on it unchanged.

**Why it exists: it is a positive control.** The measured result on real TDBRAIN
is that rTMS response is at chance from a single baseline recording. That leaves
an unanswerable question — is the *pipeline* broken, or is the *data* genuinely
uninformative? A simulator with a tunable ``effect_size`` answers it:

* ``effect_size=0`` reproduces the real null. If the pipeline still reports
  chance, it is not manufacturing signal.
* ``effect_size>0`` injects a known neurophysiological effect. If the pipeline
  now detects it, the null on real data is a property of the data, not a bug.

**What carries signal, and what does not.** Calibrated from the real cohort:
age differs between responders and non-responders (42.5 vs 46.8 years) and gender
differs slightly, so those are reproduced *with* their real association. Baseline
BDI-II does **not** separate the groups (30.8 vs 32.1) and protocol does not
either, so those are generated independent of the label. The EEG and ECG blocks
carry label information only through ``effect_size``. The consequence is that a
clinical model on this cohort lands near the real one (~0.59 AUC, driven by age)
while the neurophysiological blocks sit at chance until an effect is dialled in —
which is exactly the real cohort's behaviour.

Calibration constants below are aggregate moments (means, standard deviations,
counts) measured on the local TDBRAIN copy. They are summary statistics, not
patient records, and are of the same nature as the figures already reported in
``docs/tdbrain.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from ..preprocessing.features import BANDS, basic_features
from .loader import LoadedDataset
from .tdbrain import TDBRAIN_CHANNELS_26

# --------------------------------------------------------------------------- #
# Calibration targets measured on the real cohort (n = 131 with a responder
# label; band power and HRV pooled over a 40-patient sample).
# --------------------------------------------------------------------------- #

TARGET_N_PATIENTS = 132
TARGET_RESPONDER_RATE = 0.634
TARGET_PROTOCOL_COUNTS = (44, 88)          # protocol 1, protocol 2

# (mean, std) per class: responders first, then non-responders.
TARGET_AGE = {1: (42.45, 13.11), 0: (46.77, 14.73)}
TARGET_GENDER_P1 = {1: 0.470, 0: 0.604}     # P(gender == 1)
TARGET_BDI_PRE = {1: (30.83, 9.36), 0: (32.10, 10.61)}
TARGET_BDI_POST = {1: (7.28, 5.68), 0: (27.42, 10.50)}
BDI_PRE_RANGE = (14.0, 58.0)

# Relative band power, pooled over channels and epochs.
TARGET_BAND_POWER: dict[str, tuple[float, float]] = {
    "delta": (0.1633, 0.0831),
    "theta": (0.1175, 0.0660),
    "alpha": (0.1337, 0.0988),
    "beta": (0.1618, 0.0862),
    "gamma": (0.0652, 0.0619),
}

# Heart-rate variability, per patient.
TARGET_HR_MEAN = (69.90, 7.32)
TARGET_SDNN_MS = (41.44, 18.07)


@dataclass
class MatchedSimConfig:
    """Shape and statistics of the generated cohort.

    Defaults reproduce the real TDBRAIN cohort. ``effect_size`` is the only knob
    that injects label information into the neurophysiological blocks; at 0 the
    EEG and ECG are independent of the outcome, as measured on real data.
    """

    n_patients: int = TARGET_N_PATIENTS
    n_epochs: int = 8
    window: int = 2000                      # 8 s at 250 Hz
    fs: float = 250.0
    channels: tuple[str, ...] = TDBRAIN_CHANNELS_26
    n_rr: int = 64
    responder_rate: float = TARGET_RESPONDER_RATE
    protocol_counts: tuple[int, int] = TARGET_PROTOCOL_COUNTS

    # 0.0 = no neurophysiological signal (reproduces the real null).
    # ~0.5 = a clear, easily detectable effect. Expressed in units of the
    # between-patient standard deviation of the affected feature.
    effect_size: float = 0.0

    # Which blocks the effect acts on, when effect_size > 0.
    effect_on_eeg: bool = True
    effect_on_ecg: bool = True
    # Responders get relatively more alpha at these sites (the reference study's
    # frontal-alpha hypothesis); anything not listed is left unaffected.
    effect_channels: tuple[str, ...] = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8")

    seed: int = 42
    metadata_extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# EEG synthesis
# --------------------------------------------------------------------------- #


def _shaped_epoch(
    band_fracs: dict[str, float], window: int, fs: float, rng: np.random.Generator
) -> np.ndarray:
    """One epoch whose relative band powers match ``band_fracs``.

    Built by shaping a white-noise spectrum rather than summing sinusoids: the
    features the model consumes are *relative band powers* computed by Welch, so
    controlling the power spectral density directly is both exact and cheap.
    Random phase keeps every epoch distinct while preserving the spectrum.
    """
    n_freq = window // 2 + 1
    freqs = np.fft.rfftfreq(window, d=1.0 / fs)
    psd = np.zeros(n_freq, dtype=np.float64)

    in_band = np.zeros(n_freq, dtype=bool)
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        width = max(int(mask.sum()), 1)
        # Constant density inside the band -> integral proportional to the target.
        psd[mask] = max(band_fracs.get(name, 0.0), 1e-9) / width
        in_band |= mask

    # The five band fractions measured on real EEG sum to ~0.64, not 1: the rest of
    # the power sits below 1 Hz (drift) and above 45 Hz. That residual must be
    # generated too — `basic_features` divides each band by the total power across
    # the whole spectrum, so omitting it would renormalise every band ~1.56x up.
    residual = max(0.0, 1.0 - float(sum(band_fracs.values())))
    out_band = ~in_band
    n_out = int(out_band.sum())
    if n_out and residual > 0:
        psd[out_band] = residual / n_out

    amp = np.sqrt(psd)
    phase = rng.uniform(0, 2 * np.pi, n_freq)
    spec = amp * np.exp(1j * phase)
    spec[0] = 0.0                                    # no DC
    sig = np.fft.irfft(spec, n=window)
    # Scale to a plausible microvolt range; relative powers are scale-invariant.
    return (sig / (np.std(sig) or 1.0) * 20.0).astype(np.float32)


@lru_cache(maxsize=8)
def _spectral_corrections(window: int, fs: float) -> tuple[float, ...]:
    """Per-band factors that make the *measured* band powers hit the targets.

    Shaping the PSD analytically is not enough: ``basic_features`` measures with
    Welch (``nperseg`` 256), whose smoothing and spectral leakage bias narrow
    low-frequency bands — delta spans barely three bins and loses power to the
    sub-1 Hz region. Rather than model that analytically, the generator closes
    the loop: synthesise, measure with the very function the model will use, and
    solve for the correction by fixed-point iteration.

    Cached per ``(window, fs)`` because the bias depends on both.
    """
    rng = np.random.default_rng(0)
    corr = {b: 1.0 for b in BANDS}
    targets = {b: TARGET_BAND_POWER[b][0] for b in BANDS}

    for _ in range(6):
        epochs = [
            _shaped_epoch({b: targets[b] * corr[b] for b in BANDS}, window, fs, rng)
            for _ in range(24)
        ]
        measured = {b: 0.0 for b in BANDS}
        for ep in epochs:
            feats = basic_features(ep, fs)
            for b in BANDS:
                measured[b] += feats[f"power_{b}"] / len(epochs)
        for b in BANDS:
            if measured[b] > 1e-9:
                corr[b] *= targets[b] / measured[b]
            corr[b] = float(np.clip(corr[b], 0.05, 20.0))

    return tuple(corr[b] for b in BANDS)


def _band_fracs_for(
    label: int, channel: str, cfg: MatchedSimConfig, rng: np.random.Generator
) -> dict[str, float]:
    """Draw this epoch's target band profile, applying the effect if configured."""
    corr = _spectral_corrections(cfg.window, cfg.fs)
    fracs = {}
    for i, (name, (mu, sd)) in enumerate(TARGET_BAND_POWER.items()):
        val = rng.normal(mu, sd) * corr[i]
        fracs[name] = float(np.clip(val, 0.005, 0.85))

    if (
        cfg.effect_size > 0
        and cfg.effect_on_eeg
        and label == 1
        and channel in cfg.effect_channels
    ):
        # Responders gain alpha at the affected sites, in units of the measured
        # between-patient sd, taken from the other bands so the profile still sums
        # sensibly and the effect cannot be read off total power alone.
        mu_a, sd_a = TARGET_BAND_POWER["alpha"]
        bump = cfg.effect_size * sd_a
        fracs["alpha"] = float(np.clip(fracs["alpha"] + bump, 0.005, 0.85))
        for other in ("delta", "theta", "beta"):
            fracs[other] = float(np.clip(fracs[other] - bump / 3.0, 0.005, 0.85))
    return fracs


# --------------------------------------------------------------------------- #
# ECG synthesis
# --------------------------------------------------------------------------- #


def _tachogram_for(
    label: int, cfg: MatchedSimConfig, rng: np.random.Generator
) -> np.ndarray:
    """An RR series (seconds) with realistic mean rate and variability."""
    hr = rng.normal(*TARGET_HR_MEAN)
    sdnn_ms = abs(rng.normal(*TARGET_SDNN_MS))

    if cfg.effect_size > 0 and cfg.effect_on_ecg and label == 1:
        # Responders show higher variability — the autonomic direction reported in
        # the depression/HRV literature.
        sdnn_ms += cfg.effect_size * TARGET_SDNN_MS[1]

    hr = float(np.clip(hr, 45.0, 110.0))
    mean_rr = 60.0 / hr
    rr = rng.normal(mean_rr, sdnn_ms / 1000.0, cfg.n_rr)
    return np.clip(rr, 0.35, 1.8).astype(np.float32)


# --------------------------------------------------------------------------- #
# Clinical / demographic synthesis
# --------------------------------------------------------------------------- #


def _clinical_row(
    pid: str, label: int, protocol: int, rng: np.random.Generator
) -> dict:
    """Demographics and BDI-II scores consistent with the responder label.

    ``bdi_post`` is derived from a drawn reduction fraction rather than sampled
    independently, so the >=50% responder rule and the stored label can never
    disagree — a mismatch there would silently corrupt every downstream metric.
    """
    age = float(np.clip(rng.normal(*TARGET_AGE[label]), 18.0, 85.0))
    gender = int(rng.random() < TARGET_GENDER_P1[label])

    pre_mu, pre_sd = TARGET_BDI_PRE[label]
    bdi_pre = float(np.clip(rng.normal(pre_mu, pre_sd), *BDI_PRE_RANGE))

    post_mu, post_sd = TARGET_BDI_POST[label]
    bdi_post = float(np.clip(rng.normal(post_mu, post_sd), 0.0, bdi_pre * 1.2))
    pct = (bdi_pre - bdi_post) / bdi_pre

    # Enforce consistency with the label rather than resampling blindly.
    if label == 1 and pct < 0.5:
        bdi_post = bdi_pre * (1.0 - float(rng.uniform(0.50, 0.95)))
    elif label == 0 and pct >= 0.5:
        bdi_post = bdi_pre * (1.0 - float(rng.uniform(0.0, 0.49)))
    bdi_post = float(max(0.0, bdi_post))
    pct = (bdi_pre - bdi_post) / bdi_pre

    return {
        "patient_id": pid,
        "protocol": protocol,
        "bdi_pre": bdi_pre,
        "bdi_post": bdi_post,
        "delta_bdi": bdi_pre - bdi_post,
        "pct_reduction": float(pct),
        "responder": int(pct >= 0.5),
        "age": age,
        "gender": gender,
    }


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def simulate_matched(cfg: MatchedSimConfig | None = None) -> LoadedDataset:
    """Generate a TDBRAIN-shaped cohort as a :class:`LoadedDataset`.

    The returned object is interchangeable with :func:`load_tdbrain`'s output:
    same ``signals_mc`` layout, same ``channels``, same ``ecg`` tachogram, and a
    ``metadata`` frame carrying the columns the seeder and the clinical model
    need (``protocol``, ``bdi_pre``, ``bdi_post``, ``age``, ``gender``, ...).
    """
    cfg = cfg or MatchedSimConfig()
    rng = np.random.default_rng(cfg.seed)
    n_ch = len(cfg.channels)

    n_resp = int(round(cfg.n_patients * cfg.responder_rate))
    labels = np.array([1] * n_resp + [0] * (cfg.n_patients - n_resp), dtype=np.int8)
    rng.shuffle(labels)

    # Protocol is assigned independently of the label: on real data the two are
    # unrelated (chi2 p = 0.885), and reproducing that keeps the clinical model
    # honest about what protocol is worth.
    n1, n2 = cfg.protocol_counts
    scale = cfg.n_patients / float(n1 + n2)
    protocols = np.array(
        [1] * int(round(n1 * scale)) + [2] * (cfg.n_patients - int(round(n1 * scale))),
        dtype=np.int64,
    )
    rng.shuffle(protocols)

    signals_mc = np.empty(
        (cfg.n_patients, cfg.n_epochs, n_ch, cfg.window), dtype=np.float32
    )
    ecg = np.empty((cfg.n_patients, cfg.n_epochs, cfg.n_rr), dtype=np.float32)
    rows: list[dict] = []

    for p in range(cfg.n_patients):
        label = int(labels[p])
        pid = f"S{p:03d}"
        rows.append(_clinical_row(pid, label, int(protocols[p]), rng))

        for e in range(cfg.n_epochs):
            for c, canal in enumerate(cfg.channels):
                fracs = _band_fracs_for(label, canal, cfg, rng)
                signals_mc[p, e, c] = _shaped_epoch(fracs, cfg.window, cfg.fs, rng)

        # HRV is a patient-level trait: one tachogram, repeated across epochs —
        # identical to how the real loader stores it.
        tach = _tachogram_for(label, cfg, rng)
        ecg[p, :, :] = tach

    rep_idx = cfg.channels.index("Pz") if "Pz" in cfg.channels else 0
    metadata = pd.DataFrame(rows)

    return LoadedDataset(
        signals=np.ascontiguousarray(signals_mc[:, :, rep_idx, :]),
        labels=np.asarray([r["responder"] for r in rows], dtype=np.int8),
        fs=cfg.fs,
        window=cfg.window,
        metadata=metadata,
        erp=None,
        ecg=ecg,
        channels=list(cfg.channels),
        signals_mc=signals_mc,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate the TDBRAIN-matched cohort.")
    ap.add_argument("--effect", type=float, default=0.0,
                    help="effect size injected into EEG/ECG (0 = reproduce the real null)")
    ap.add_argument("--patients", type=int, default=TARGET_N_PATIENTS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = simulate_matched(MatchedSimConfig(
        n_patients=args.patients, effect_size=args.effect, seed=args.seed
    ))
    print(f"patients={ds.signals_mc.shape[0]} epochs={ds.signals_mc.shape[1]} "
          f"channels={ds.signals_mc.shape[2]} window={ds.window} fs={ds.fs}")
    print(f"responders={int(ds.labels.sum())}/{len(ds.labels)} "
          f"({ds.labels.mean():.3f})")
    print(f"ecg={None if ds.ecg is None else ds.ecg.shape}")
    print(ds.metadata.describe().round(2).to_string())
