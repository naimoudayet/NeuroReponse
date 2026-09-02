"""The ablation ladder the new_docs hypotheses ask for, run and scored honestly.

``new_docs/HYPO1_HYPO4_Tesla_369_rTMS_EEG_ECG_LSTM.docx`` section 11 specifies an
experimental protocol — models A to E, patient-wise splits, ROC-AUC / PR-AUC /
balanced accuracy / F1 / calibration, and ablations that quantify each layer's
contribution separately. ``new_docs/Guide_Utilisateur_perfectionne_equations_RES0_AR1``
section 5.8 specifies the same ladder with the LSTM equations written out. This
module *is* that protocol, executed:

    python -m src.reporting.hypo_ablation --db recherche_tdbrain.sqlite3

Two things it does that the reference study (Arteaga et al., PMC12981298) does
not, and which are the point of running it at all:

1. **It tests feature families the study never computed.** Their model, and this
   project's original TDBRAIN arm, both see relative band power per channel and
   nothing else. Band power is blind to phase, so no amount of it can express
   the Kuramoto coupling, the PLV or the coherence the documents put at the
   centre of the model. Those blocks now exist (``src.preprocessing.connectivity``)
   and get their own rungs on the ladder.

2. **It applies a stopping rule that a null result can fail.**
   :func:`src.models.metrics.beats_chance` requires the AUC confidence interval
   to exclude 0.5, the permutation p to clear alpha, balanced accuracy above 0.5
   *and* the model to have predicted both classes. On a cohort with a 83/132
   base rate, accuracy and F1 alone rate the all-positive predictor as a working
   model — which is exactly the failure this project already measured across all
   four variants of its 2x2.

**Model E of the document's ladder is absent on purpose, and that absence is a
result.** See :func:`physics_proxy`.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..data.loader import LoadedDataset
from ..data.modalities import MODALITY_ORDER, build_features, protocol_mask
from ..models.lstm import LSTMConfig
from ..models.metrics import (
    N_PERMUTATIONS,
    beats_chance,
    classification_report_full,
)
from ..models.train import TrainConfig, cross_validate

OUT_PATH = Path("data/models/hypo_ablation.json")

# The ladder. Keys A-D are the document's own model names; the rest are the rungs
# it could not specify because the feature blocks did not exist yet.
#
# Every rung differs from the one above it by exactly one block, so the delta is
# attributable. That is what makes it an ablation rather than a leaderboard.
LADDER: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("A",      "EEG seul (article)",          ("eeg",)),
    ("B",      "ECG seul (HRV)",              ("ecg",)),
    ("C",      "EEG + ECG",                   ("eeg", "ecg")),
    ("D",      "EEG + ECG + clinique",        ("rtms", "eeg", "ecg")),
    ("SYNC",   "Synchronisation seule",       ("sync",)),
    ("CPLX",   "Complexité seule",            ("cplx",)),
    ("NET",    "Réseau (sync + cplx)",        ("sync", "cplx")),
    ("CLIN",   "Clinique seul (référence)",   ("rtms",)),
    ("F",      "Clinique + réseau",           ("rtms", "sync", "cplx")),
    ("G",      "Tout sauf 3-6-9",             ("rtms", "eeg", "ecg", "sync", "cplx")),
    ("H369",   "Tout + 3-6-9 (HYPO4-369)",    ("rtms", "eeg", "ecg", "sync", "cplx", "h369")),
)


@dataclass
class Rung:
    """One model of the ladder, with every number the guide's 5.9 asks for."""

    model: str
    label: str
    modalities: list[str]
    n_patients: int
    n_features: int
    accuracy: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    auc: float
    auc_ci_lo: float
    auc_ci_hi: float
    pr_auc: float
    pr_auc_baseline: float
    brier: float
    brier_baseline: float
    p_perm_auc: float
    base_rate: float
    predicted_positive_rate: float
    auc_fold_mean: float
    auc_fold_std: float
    beats_chance: bool
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Model E — the electromagnetic layer, and why it cannot be built here
# --------------------------------------------------------------------------- #

def physics_proxy(dataset: LoadedDataset) -> tuple[np.ndarray, tuple[str, ...]]:
    """The best B/E/J block TDBRAIN's published metadata can support. It is rank 1.

    HYPO4 layer 1 asks for Biot-Savart ``B(r,t) = (mu0/4pi) I(t) ∮ dl x r / |r|^3``,
    Faraday ``∇xE = -∂B/∂t`` and Ohm ``J = sigma E``. Follow the chain through and
    see what each term needs from the data:

    * ``I(t)`` — coil current. Set by the stimulator intensity, which TDBRAIN
      **does not publish**: this project stores it as ``0.0`` with a protocol
      string saying so, and ``total_pulses()`` is 0 for every patient.
    * the coil integral — needs the coil geometry and its position on the head.
      Not published either.
    * ``∂B/∂t`` — scales with the pulse waveform and the repetition rate. The
      rate *is* published: 10 Hz for protocol 1, 1 Hz for protocol 2.
    * ``sigma(r)`` — tissue conductivity map, which needs an individual MRI.
      TDBRAIN ships no imaging.

    So every physically derived quantity reduces to an unknown constant times a
    function of the protocol integer, and the protocol integer takes two values.
    This function builds that block explicitly — induced field amplitude
    ``prop. f``, energy ``prop. f^2``, and the excitatory/inhibitory sign of the
    protocol — so the claim can be *checked* rather than asserted: all three
    columns are perfectly determined by ``protocol``, which the clinical block
    already contains.

    The project has separately measured what that one bit is worth: responder
    rate 61.4% versus 64.4% across the two arms, chi-2 p = 0.885, AUC 0.514,
    mutual information 0.0004 nats. Model E of the ladder is therefore not
    "unimplemented" — it is provably a re-encoding of a column already present,
    carrying no information. Adding it would only widen the vector.
    """
    protocol = dataset.metadata["protocol"].to_numpy(dtype=np.float64)
    freq = np.where(protocol == 1, 10.0, 1.0)          # published repetition rate
    return (
        np.column_stack([
            freq,                                       # |E| prop. dB/dt prop. f
            freq ** 2,                                  # deposited energy prop. f^2
            np.where(protocol == 1, 1.0, -1.0),         # excitatory vs inhibitory
        ]),
        ("e_field_amplitude", "e_field_energy", "protocol_sign"),
    )


def physics_is_collinear(dataset: LoadedDataset) -> dict[str, float]:
    """Numerical proof that :func:`physics_proxy` adds no column of its own.

    Returns the rank of ``[protocol | physics]`` against the rank of
    ``[protocol]``. Equal ranks mean every physics column lies in the span of the
    protocol indicator: the electromagnetic layer is a relabelling.
    """
    protocol = dataset.metadata["protocol"].to_numpy(dtype=np.float64)[:, None]
    phys, _ = physics_proxy(dataset)
    ones = np.ones_like(protocol)
    base = np.hstack([ones, protocol])
    joint = np.hstack([base, phys])
    return {
        "rank_protocol": float(np.linalg.matrix_rank(base)),
        "rank_protocol_plus_physics": float(np.linalg.matrix_rank(joint)),
        "n_physics_columns": float(phys.shape[1]),
        "distinct_protocol_values": float(np.unique(protocol).size),
    }


# --------------------------------------------------------------------------- #
# Feature assembly, built once and reused across the ladder
# --------------------------------------------------------------------------- #

def build_all_blocks(
    dataset: LoadedDataset, per_patient_zscore: bool = True, verbose: bool = True
) -> dict[str, tuple[np.ndarray, tuple[str, ...]]]:
    """Every modality block, computed once.

    The ladder asks for eleven models over six blocks; rebuilding a block per
    model would recompute the 30 synchronisation features 1056 times more than
    necessary. Blocks are keyed by modality name and reassembled in
    :data:`src.data.modalities.MODALITY_ORDER` — the *same* constant
    ``build_features`` uses, so a rung can never receive a differently ordered
    vector than the single-call path would have produced.
    ``test_ladder_assembly_matches_build_features`` pins that equality.
    """
    from ..data.modalities import (
        complexity_block,
        ecg_block,
        eeg_block,
        h369_block,
        rtms_block,
        sync_block,
    )

    mc = dataset.signals_mc
    n_epochs = mc.shape[1] if mc is not None else dataset.signals.shape[1]
    builders = {
        "rtms": lambda: rtms_block(dataset, n_epochs),
        "eeg": lambda: eeg_block(dataset, per_patient_zscore),
        "ecg": lambda: ecg_block(dataset),
        "sync": lambda: sync_block(dataset),
        "cplx": lambda: complexity_block(dataset),
        "h369": lambda: h369_block(dataset),
    }
    out: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {}
    for name, fn in builders.items():
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[name] = fn()
        if verbose:
            print(f"  block {name:<5} {out[name][0].shape[-1]:>4} features "
                  f"({time.time() - t0:5.1f}s)", flush=True)
    return out


def assemble(
    blocks: dict[str, tuple[np.ndarray, tuple[str, ...]]], modalities: tuple[str, ...]
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Concatenate cached blocks in canonical order."""
    chosen = [m for m in MODALITY_ORDER if m in modalities]
    x = np.concatenate([blocks[m][0] for m in chosen], axis=-1)
    names = tuple(n for m in chosen for n in blocks[m][1])
    return x.astype(np.float32), names


# --------------------------------------------------------------------------- #
# Running the ladder
# --------------------------------------------------------------------------- #

def evaluate_rung(
    model: str,
    label: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    modalities: tuple[str, ...],
    n_splits: int = 5,
    repeats: int = 1,
    train_cfg: TrainConfig | None = None,
    n_permutations: int = N_PERMUTATIONS,
    n_boot: int = 1000,
) -> Rung:
    """Patient-wise CV for one rung, scored with the full metric suite."""
    t0 = time.time()
    train_cfg = train_cfg or TrainConfig()
    cv = cross_validate(
        x, y.astype(np.float32), groups,
        lstm_cfg=LSTMConfig(input_size=x.shape[-1]),
        train_cfg=train_cfg, n_splits=n_splits, repeats=repeats,
        standardise=True,
    )
    y_true, y_proba = cv.out_of_fold(y)
    report = classification_report_full(
        y_true, y_proba, n_permutations=n_permutations, n_boot=n_boot,
        seed=train_cfg.seed,
    )
    summary = cv.summary()
    return Rung(
        model=model, label=label, modalities=list(modalities),
        n_patients=int(x.shape[0]), n_features=int(x.shape[-1]),
        accuracy=report["accuracy"],
        balanced_accuracy=report["balanced_accuracy"],
        sensitivity=report["sensitivity"], specificity=report["specificity"],
        precision=report["precision"], f1=report["f1"],
        auc=report["auc"], auc_ci_lo=report["auc_ci_lo"],
        auc_ci_hi=report["auc_ci_hi"],
        pr_auc=report["pr_auc"], pr_auc_baseline=report["pr_auc_baseline"],
        brier=report["brier"], brier_baseline=report["brier_baseline"],
        p_perm_auc=report["p_perm_auc"], base_rate=report["base_rate"],
        predicted_positive_rate=report["predicted_positive_rate"],
        auc_fold_mean=float(summary["auc_mean"]),
        auc_fold_std=float(summary["auc_std"]),
        beats_chance=beats_chance(report),
        seconds=float(time.time() - t0),
    )


def run_ladder(
    dataset: LoadedDataset,
    protocol: int | None = None,
    n_splits: int = 5,
    repeats: int = 1,
    train_cfg: TrainConfig | None = None,
    ladder=LADDER,
    blocks=None,
    verbose: bool = True,
) -> list[Rung]:
    """Every rung, from one pass of feature extraction."""
    if blocks is None:
        if verbose:
            print("extracting feature blocks ...", flush=True)
        blocks = build_all_blocks(dataset, verbose=verbose)

    _, y, groups, _ = build_features(
        dataset, modalities=("rtms",), target="responder", protocol=None
    )
    mask = protocol_mask(dataset, protocol)

    rows: list[Rung] = []
    for model, label, modalities in ladder:
        x, _names = assemble(blocks, modalities)
        row = evaluate_rung(
            model, label, x[mask], y[mask], groups[mask], modalities,
            n_splits=n_splits, repeats=repeats, train_cfg=train_cfg,
        )
        if verbose:
            verdict = "SIGNAL" if row.beats_chance else "-"
            print(
                f"  {model:<6}{label:<30}{row.n_features:>5} feat  "
                f"AUC {row.auc:.3f} [{row.auc_ci_lo:.3f},{row.auc_ci_hi:.3f}]  "
                f"bal.acc {row.balanced_accuracy:.3f}  "
                f"p={row.p_perm_auc:.3f}  {verdict}",
                flush=True,
            )
        rows.append(row)
    return rows


def shuffled_label_control(
    dataset: LoadedDataset,
    modalities: tuple[str, ...],
    n_shuffles: int = 3,
    protocol: int | None = None,
    n_splits: int = 5,
    train_cfg: TrainConfig | None = None,
    blocks=None,
) -> list[float]:
    """Retrain the whole pipeline on permuted labels; return the AUCs it reaches.

    **A permutation p-value is not this test.** ``permutation_p_auc`` shuffles
    labels after the predictions exist, so it only nulls the statistic — it
    reported p = 0.010 for this project's early-stopping leak, numbers that were
    entirely artefactual. Only retraining end to end on permuted targets can
    catch a leak upstream of the prediction, and CLAUDE.md makes clearing it a
    precondition for any new claim of signal. Labels are permuted **per patient**,
    matching the group structure the splitter uses.
    """
    blocks = blocks if blocks is not None else build_all_blocks(dataset, verbose=False)
    x, _ = assemble(blocks, modalities)
    _, y, groups, _ = build_features(dataset, modalities=("rtms",), target="responder")
    mask = protocol_mask(dataset, protocol)
    x, y, groups = x[mask], y[mask], groups[mask]

    train_cfg = train_cfg or TrainConfig()
    out: list[float] = []
    for k in range(n_shuffles):
        rng = np.random.default_rng(1000 + k)
        y_shuffled = rng.permutation(y)
        cv = cross_validate(
            x, y_shuffled.astype(np.float32), groups,
            lstm_cfg=LSTMConfig(input_size=x.shape[-1]),
            train_cfg=train_cfg, n_splits=n_splits, standardise=True,
        )
        yt, yp = cv.out_of_fold(y_shuffled)
        from sklearn.metrics import roc_auc_score
        out.append(float(roc_auc_score(yt, yp)))
    return out


# --------------------------------------------------------------------------- #
# Positive control — do the new features measure anything real?
# --------------------------------------------------------------------------- #
#
# A null result is only worth reporting if the instrument works. "We computed
# connectivity and it predicted nothing" and "we computed connectivity wrongly"
# produce identical tables, so the ladder alone cannot distinguish them. This
# control does, by checking the new features against relationships that are
# established in the EEG literature and have nothing to do with rTMS response.
#
# Each entry is (feature, expected sign of its correlation with age, source):
#   * the 1/f aperiodic exponent flattens with age (Voytek et al. 2015), so the
#     exponent as defined here — positive = steeper — must fall;
#   * individual alpha frequency declines with age;
#   * alpha-band phase synchrony declines with age.
# All three are large, replicated effects. If they fail to appear, the extraction
# is broken and no conclusion about response can be drawn from it.
KNOWN_AGE_EFFECTS: tuple[tuple[str, int, str], ...] = (
    ("aperiodic_exponent_mean", -1, "1/f flattening with age (Voytek 2015)"),
    ("alpha_peak_hz", -1, "individual alpha frequency declines with age"),
    ("plv_mean_alpha", -1, "alpha-band phase synchrony declines with age"),
)


def feature_validity_report(
    dataset: LoadedDataset, blocks=None, alpha: float = 0.05
) -> dict:
    """Two controls the ladder cannot supply on its own.

    **Positive control.** Correlate the new features with age and check the three
    :data:`KNOWN_AGE_EFFECTS` come out with the right sign and a significant p.
    Age is the ideal probe here: it is measured exactly, it is known to be encoded
    in exactly these quantities, and it is not the outcome under test.

    **Univariate screen with FDR.** Correlate every new feature with every
    outcome and apply Benjamini-Hochberg. HYPO4-369 requires this explicitly —
    its own falsification criterion is that the 3-6-9 index must survive
    multiple-comparison correction. With 40 features and three outcomes, two
    nominal hits below 0.05 are the *expectation* under the null, so an
    uncorrected screen would find "signal" in pure noise every time.
    """
    from scipy.stats import pearsonr

    from ..data.modalities import target_values
    from ..models.metrics import benjamini_hochberg

    blocks = blocks if blocks is not None else build_all_blocks(dataset, verbose=False)
    # Patient-level values: the epoch axis is eight windows of one recording, so
    # averaging over it is the patient's own estimate, not a loss of information.
    x = np.concatenate(
        [blocks[m][0].mean(axis=1) for m in ("sync", "cplx", "h369")], axis=-1
    )
    names = tuple(n for m in ("sync", "cplx", "h369") for n in blocks[m][1])
    age = dataset.metadata["age"].to_numpy(dtype=np.float64)

    def _screen(target: np.ndarray) -> dict:
        rs, ps = [], []
        for i in range(x.shape[1]):
            col = x[:, i]
            if np.std(col) < 1e-12 or not np.isfinite(col).all():
                rs.append(np.nan)
                ps.append(np.nan)
                continue
            r, p = pearsonr(col, target)
            rs.append(float(r))
            ps.append(float(p))
        rejected, q = benjamini_hochberg(ps, alpha)
        finite = np.isfinite(ps)
        best = int(np.nanargmin(np.where(finite, ps, np.nan))) if finite.any() else 0
        return {
            "n_tested": int(x.shape[1]),
            "n_nominal_hits": int(np.nansum(np.asarray(ps) < alpha)),
            "n_survivors_fdr": int(rejected.sum()),
            "best_feature": names[best],
            "best_r": float(rs[best]),
            "best_p": float(ps[best]),
            "best_q": float(q[best]),
        }

    controls = []
    for feature, expected_sign, source in KNOWN_AGE_EFFECTS:
        i = names.index(feature)
        r, p = pearsonr(x[:, i], age)
        controls.append({
            "feature": feature, "source": source,
            "r_age": float(r), "p": float(p),
            "expected_sign": expected_sign,
            "confirmed": bool(np.sign(r) == expected_sign and p < alpha),
        })

    screens = {"responder": _screen(dataset.labels.astype(np.float64))}
    for target in ("delta_bdi", "pct_reduction"):
        try:
            screens[target] = _screen(target_values(dataset, target))
        except Exception:                     # target absent from this cohort
            continue

    return {
        "age_controls": controls,
        "all_age_controls_confirmed": all(c["confirmed"] for c in controls),
        "screens": screens,
    }


# --------------------------------------------------------------------------- #
# The regression ladder — the arm that is directly comparable with the article
# --------------------------------------------------------------------------- #
#
# The binary ladder above answers "is there a responder signal?". It cannot be
# put next to Arteaga et al.'s headline number, because they regress a continuous
# BDI-II change and report Pearson r; r and AUC are not comparable quantities.
# This ladder runs the *same* feature sets against their target, per protocol, so
# the comparison is like-for-like — and it carries the baseline the article never
# reports, which is the whole reason the comparison is worth making.

REG_LADDER: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("A",    "EEG seul (article)",       ("eeg",)),
    ("CLIN", "Clinique seul (référence)", ("rtms",)),
    ("NET",  "Réseau (sync + cplx)",     ("sync", "cplx")),
    ("A+NET", "EEG + réseau",            ("eeg", "sync", "cplx")),
    ("G",    "Tout sauf 3-6-9",          ("rtms", "eeg", "ecg", "sync", "cplx")),
    ("H369", "Tout + 3-6-9",             ("rtms", "eeg", "ecg", "sync", "cplx", "h369")),
)

# Arteaga et al. (PMC12981298), Table 3 — the number this project's regression
# arm is measured against. Reproduced here so the comparison is in the artefact
# rather than in a reader's memory.
ARTICLE_R = 0.401


@dataclass
class RegRung:
    model: str
    label: str
    modalities: list[str]
    protocol: int | None
    target: str
    n_patients: int
    n_features: int
    r_oof: float
    r_mean: float
    r_std: float
    p_perm: float
    r2: float
    mae: float
    rmse: float
    r_partial: float
    pred_sd: float
    target_sd: float
    baseline_r_bdi_pre: float
    beats_baseline: bool
    beats_article: bool
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_reg_rung(
    model: str,
    label: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    modalities: tuple[str, ...],
    protocol: int | None,
    target: str,
    bdi_pre: np.ndarray,
    n_splits: int = 5,
    repeats: int = 1,
    train_cfg: TrainConfig | None = None,
    n_permutations: int = N_PERMUTATIONS,
) -> RegRung:
    """One regression rung, scored against the trivial baseline it must clear."""
    from ..models.lstm import REGRESSION
    from ..models.metrics import partial_correlation, pearson_r, regression_report

    t0 = time.time()
    train_cfg = train_cfg or TrainConfig()
    cv = cross_validate(
        x, y.astype(np.float32), groups,
        lstm_cfg=LSTMConfig(input_size=x.shape[-1], task=REGRESSION),
        train_cfg=train_cfg, n_splits=n_splits, repeats=repeats, standardise=True,
    )
    y_true, y_pred = cv.out_of_fold(y)
    pooled = regression_report(y_true, y_pred, n_permutations=n_permutations)
    summary = cv.summary()
    base = pearson_r(y_true, bdi_pre)
    return RegRung(
        model=model, label=label, modalities=list(modalities), protocol=protocol,
        target=target, n_patients=int(x.shape[0]), n_features=int(x.shape[-1]),
        r_oof=float(pooled["r"]), r_mean=float(summary["r_mean"]),
        r_std=float(summary["r_std"]), p_perm=float(pooled["p_perm"]),
        r2=float(pooled["r2"]), mae=float(pooled["mae"]), rmse=float(pooled["rmse"]),
        r_partial=float(partial_correlation(y_true, y_pred, bdi_pre)),
        # A near-zero pred_sd against a large target_sd is this project's
        # signature failure: the model emits the cohort mean, and any r it still
        # shows is a ranking of noise. Reported next to r2 for that reason.
        pred_sd=float(np.std(y_pred)), target_sd=float(np.std(y_true)),
        baseline_r_bdi_pre=float(base),
        beats_baseline=bool(
            np.isfinite(pooled["r"]) and np.isfinite(base) and pooled["r"] > base
        ),
        beats_article=bool(np.isfinite(pooled["r"]) and pooled["r"] > ARTICLE_R),
        seconds=float(time.time() - t0),
    )


def run_regression_ladder(
    dataset: LoadedDataset,
    protocols: tuple[int | None, ...] = (1, 2),
    targets: tuple[str, ...] = ("delta_bdi", "pct_reduction"),
    n_splits: int = 5,
    repeats: int = 1,
    train_cfg: TrainConfig | None = None,
    ladder=REG_LADDER,
    blocks=None,
    verbose: bool = True,
) -> list[RegRung]:
    """The regression ladder, per protocol and per target, from one extraction."""
    from ..data.modalities import target_values

    blocks = blocks if blocks is not None else build_all_blocks(dataset, verbose=verbose)
    _, _, groups_all, _ = build_features(dataset, modalities=("rtms",), target="responder")
    bdi_all = dataset.metadata["bdi_pre"].to_numpy(dtype=np.float64)

    rows: list[RegRung] = []
    for protocol in protocols:
        mask = protocol_mask(dataset, protocol)
        for target in targets:
            y_all = target_values(dataset, target)
            for model, label, modalities in ladder:
                x, _ = assemble(blocks, modalities)
                row = evaluate_reg_rung(
                    model, label, x[mask], y_all[mask], groups_all[mask],
                    modalities, protocol, target, bdi_all[mask],
                    n_splits=n_splits, repeats=repeats, train_cfg=train_cfg,
                )
                if verbose:
                    arm = "P1+P2" if protocol is None else f"P{protocol}"
                    print(
                        f"  {arm} {target:<14}{model:<7}{row.n_features:>5} feat  "
                        f"r={row.r_oof:+.3f} (p={row.p_perm:.3f})  "
                        f"R2={row.r2:+.3f}  base={row.baseline_r_bdi_pre:+.3f}  "
                        f"{'BEATS BASE' if row.beats_baseline else '-'}",
                        flush=True,
                    )
                rows.append(row)
    return rows


def regression_table(rows: list[RegRung]) -> str:
    head = (
        f"{'arm':<7}{'target':<15}{'model':<7}{'n':>4}{'feat':>6}"
        f"{'r (OOF)':>10}{'p':>7}{'r_part':>9}{'R2':>8}"
        f"{'pred sd':>9}{'base r':>9}{'vs base':>9}{'vs art.':>9}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        arm = "P1+P2" if r.protocol is None else f"P{r.protocol}"
        lines.append(
            f"{arm:<7}{r.target:<15}{r.model:<7}{r.n_patients:>4}{r.n_features:>6}"
            f"{r.r_oof:>+10.3f}{r.p_perm:>7.3f}{r.r_partial:>+9.3f}{r.r2:>+8.3f}"
            f"{r.pred_sd:>9.2f}{r.baseline_r_bdi_pre:>+9.3f}"
            f"{('beats' if r.beats_baseline else '-'):>9}"
            f"{('beats' if r.beats_article else '-'):>9}"
        )
    return "\n".join(lines)


def ladder_table(rows: list[Rung]) -> str:
    head = (
        f"{'':<6}{'model':<30}{'feat':>5}{'AUC':>8}{'95% CI':>16}"
        f"{'PR-AUC':>8}{'bal.acc':>9}{'spec':>7}{'Brier':>8}{'p':>7}  verdict"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r.model:<6}{r.label:<30}{r.n_features:>5}{r.auc:>8.3f}"
            f"{f'[{r.auc_ci_lo:.2f},{r.auc_ci_hi:.2f}]':>16}"
            f"{r.pr_auc:>8.3f}{r.balanced_accuracy:>9.3f}{r.specificity:>7.3f}"
            f"{r.brier:>8.3f}{r.p_perm_auc:>7.3f}  "
            f"{'SIGNAL' if r.beats_chance else '-'}"
        )
    if rows:
        r = rows[0]
        lines.append("-" * len(head))
        lines.append(
            f"{'':<6}{'no-skill reference':<30}{'':>5}{0.5:>8.3f}{'':>16}"
            f"{r.pr_auc_baseline:>8.3f}{0.5:>9.3f}{0.0:>7.3f}"
            f"{r.brier_baseline:>8.3f}"
        )
    return "\n".join(lines)


def read_json(path: Path = OUT_PATH) -> dict:
    """The ladder as the app reads it; empty when it has never been run."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the HYPO4 / RES0_AR1 ablation ladder on a seeded cohort."
    )
    ap.add_argument("--db", type=Path, default=Path("recherche_tdbrain.sqlite3"))
    ap.add_argument("--protocol", type=int, default=None, choices=(1, 2))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffles", type=int, default=3,
                    help="end-to-end retrains on permuted labels for the best rung")
    ap.add_argument("--regression", action="store_true",
                    help="also run the continuous-target ladder (the article's arm)")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    from ..data.tdbrain_seeder import dataset_from_repository
    from ..db import Repository

    print(f"loading {args.db} ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = dataset_from_repository(Repository(db_url=f"sqlite:///{args.db}"))

    train_cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
    print("extracting feature blocks ...", flush=True)
    blocks = build_all_blocks(dataset, verbose=True)
    rows = run_ladder(
        dataset, protocol=args.protocol, n_splits=args.n_splits,
        repeats=args.repeats, train_cfg=train_cfg, blocks=blocks,
    )

    validity = feature_validity_report(dataset, blocks=blocks)
    print("\n" + "-" * 104)
    print("CONTRÔLE POSITIF — les nouveaux features mesurent-ils quelque chose de réel ?")
    for c in validity["age_controls"]:
        print(f"  {c['feature']:<28} r(âge) = {c['r_age']:+.3f}  p = {c['p']:.2e}  "
              f"{'CONFIRMÉ' if c['confirmed'] else 'NON CONFIRMÉ'}   ({c['source']})")
    print("\nCRIBLAGE UNIVARIÉ (correction FDR de Benjamini-Hochberg, alpha = 5 %)")
    for target, s in validity["screens"].items():
        print(f"  {target:<14} meilleur = {s['best_feature']:<24} "
              f"r = {s['best_r']:+.3f}  p = {s['best_p']:.3f}  q = {s['best_q']:.3f}  "
              f"| {s['n_nominal_hits']}/{s['n_tested']} nominaux, "
              f"{s['n_survivors_fdr']} survivent au FDR")

    print("\n" + "=" * 104)
    arm = "les deux protocoles" if args.protocol is None else f"protocole {args.protocol}"
    print(f"ABLATION LADDER (HYPO4 §11 / RES0_AR1 §5.8) — cible binaire, {arm}")
    print("=" * 104)
    print(ladder_table(rows))

    collinear = physics_is_collinear(dataset)
    print(
        "\nModel E (B/E/J physique) — non calculable : rang(protocole) = "
        f"{collinear['rank_protocol']:.0f}, rang(protocole + physique) = "
        f"{collinear['rank_protocol_plus_physics']:.0f} pour "
        f"{collinear['n_physics_columns']:.0f} colonnes ajoutées. "
        "Les variables électromagnétiques sont entièrement déterminées par le "
        "protocole, déjà présent dans le bloc clinique."
    )

    best = max(rows, key=lambda r: (r.auc if np.isfinite(r.auc) else -1))
    shuffles: list[float] = []
    if args.shuffles > 0:
        print(f"\ncontrôle labels permutés sur « {best.label} » "
              f"({args.shuffles} réentraînements complets) ...", flush=True)
        shuffles = shuffled_label_control(
            dataset, tuple(best.modalities), n_shuffles=args.shuffles,
            protocol=args.protocol, n_splits=args.n_splits, train_cfg=train_cfg,
            blocks=blocks,
        )
        print(f"  AUC sur labels permutés : "
              f"{', '.join(f'{v:.3f}' for v in shuffles)}  "
              f"(réel : {best.auc:.3f})")

    reg_rows: list[RegRung] = []
    if args.regression:
        print("\nladder de régression (cible continue, comparable à l'article) ...",
              flush=True)
        reg_rows = run_regression_ladder(
            dataset, n_splits=args.n_splits, repeats=args.repeats,
            train_cfg=train_cfg, blocks=blocks,
        )
        print("\n" + "=" * 104)
        print("LADDER DE RÉGRESSION — cible continue, un modèle par protocole "
              f"(article : r = {ARTICLE_R:.3f})")
        print("=" * 104)
        print(regression_table(reg_rows))
        print(
            "\n'base r' = corrélation du BDI-II initial seul avec la cible, sans "
            "modèle. L'article ne le rapporte pas ; sur delta_bdi il dépasse son "
            f"r = {ARTICLE_R:.3f}, donc « battre l'article » et « battre la "
            "sévérité de départ » ne sont pas la même exigence — seule la "
            "seconde signifie quelque chose."
        )

    payload = {
        "protocol": args.protocol,
        "n_splits": args.n_splits,
        "repeats": args.repeats,
        "article_r": ARTICLE_R,
        "rows": [r.to_dict() for r in rows],
        "regression_rows": [r.to_dict() for r in reg_rows],
        "feature_validity": validity,
        "physics_collinearity": collinear,
        "shuffled_label_control": {
            "modalities": list(best.modalities),
            "real_auc": best.auc,
            "shuffled_aucs": shuffles,
        },
        "any_signal": bool(any(r.beats_chance for r in rows)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nrésultats -> {args.out}")


if __name__ == "__main__":
    _main()
