"""Seed the SQLite database from the **real** TDBRAIN cohort.

The sibling of :mod:`src.data.seeder` (which seeds from the simulator). Both end
up in the same `Patient` / `SessionRTMS` / `SignalNeurophysiologique` domain
objects, so the Streamlit app works against either source unchanged.

What is real here and what is a stand-in — this matters for the defence:

* **Real** — the EEG itself (26-channel montage), the BDI-II pre/post scores,
  the responder label, age/gender, and the rTMS *protocol* (frequency +
  stimulation site, which the protocol number defines).
* **Stand-in** — the "sessions". TDBRAIN records **one baseline resting EEG per
  patient**, not a treatment trajectory, so each stored session is an **epoch**
  of that single recording. Every epoch therefore carries the same date, the
  same rTMS parameters and the same BDI scores; only the EEG differs. The
  `protocole` label on each session says so explicitly, so nothing in the UI
  claims a per-session evolution that the data does not contain.
* **Unknown** — per-patient stimulation intensity, train duration and train
  count are not published with TDBRAIN. They are stored as ``0.0`` and flagged
  in the `protocole` string rather than filled with plausible-looking numbers.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..db import Repository
from ..domain import (
    Patient,
    RTMSParameters,
    SessionRTMS,
    SignalNeurophysiologique,
    SignalType,
)
from ..domain.patient import DossierClinique
from .loader import LoadedDataset

# The rTMS protocol number encodes a real, published stimulation protocol.
# (TDBRAIN data descriptor / Brainclinics rTMS-in-MDD cohort.)
PROTOCOLS: dict[int, tuple[float, str, str]] = {
    1: (10.0, "L-DLPFC", "10 Hz cortex préfrontal dorsolatéral gauche"),
    2: (1.0, "R-DLPFC", "1 Hz cortex préfrontal dorsolatéral droit"),
}

# TDBRAIN is anonymised and ships no acquisition dates. A single fixed reference
# date is used for every recording — inventing distinct dates would imply a
# chronology the dataset does not have.
REFERENCE_DATE = datetime(2020, 1, 1)


def _rtms_parameters(protocol: int | None) -> RTMSParameters:
    freq, site, human = PROTOCOLS.get(
        int(protocol) if protocol is not None else -1,
        (0.0, "inconnu", "protocole non renseigné"),
    )
    return RTMSParameters(
        frequence_hz=freq,
        intensite_pct=0.0,      # not published per patient — see module docstring
        duree_train_s=0.0,      # idem
        nb_trains=0,            # idem
        intervalle_train_s=0.0,  # idem
        localisation=site,
        protocole=(
            f"TDBRAIN — {human} · séance = époque du repos de référence "
            f"· paramètres de stimulation non publiés"
        ),
    )


def protocol_from_parameters(params: RTMSParameters) -> int | None:
    """Inverse of :func:`_rtms_parameters` — the arm number behind stored params.

    The seeder persists frequency and site rather than the protocol integer, but
    both are deterministic functions of it (see the module docstring), so the map
    inverts without ambiguity. The clinical block feeds the model that integer, so
    the inverse lives here next to the forward map: splitting them is how the two
    drift apart. Returns ``None`` when the parameters match no known protocol.
    """
    for protocol, (freq, site, _human) in PROTOCOLS.items():
        if abs(float(params.frequence_hz) - freq) < 1e-9 and params.localisation == site:
            return protocol
    return None


def _sexe(row: pd.Series) -> int | None:
    """Gender as the 0/1 integer the clinical block was trained on.

    ``load_tdbrain`` keeps it as a *string* (best-effort parsing of
    participants.tsv) while the matched simulator emits an int. Both must land on
    the same number here, because :func:`src.data.modalities.rtms_block` coerces
    the column with ``to_numpy(dtype=float64)`` and that is what the model saw.
    Anything unparseable becomes ``None`` — never 0, which is a real category.
    """
    raw = row.get("gender")
    if raw is None:
        return None
    value = pd.to_numeric(raw, errors="coerce")
    return int(value) if np.isfinite(value) else None


def _patient_from_tdbrain(
    row: pd.Series,
    epochs: np.ndarray,      # (n_epochs, n_channels, window)
    channels: list[str],
    fs: float,
    tachogram: np.ndarray | None = None,   # (n_epochs, n_rr) RR intervals in seconds
    ecg_channel: str = "Erbs",
) -> Patient:
    """Map one TDBRAIN subject onto the project's domain model."""
    subject_id = str(row["patient_id"])
    age = row.get("age")
    bdi_pre = float(row["bdi_pre"])
    bdi_post = float(row["bdi_post"])
    protocol = int(row["protocol"])
    responder = int(row["responder"])
    pct = float(row["pct_reduction"])

    freq, site, _ = PROTOCOLS.get(protocol, (0.0, "inconnu", ""))
    patient = Patient(
        id=subject_id,
        nom=f"Sujet {subject_id}",       # TDBRAIN ids are already pseudonymous
        # Stored unrounded: participants.tsv publishes decimal ages (49.66) and
        # that exact number is what fed the clinical block at training time.
        age=float(age) if age is not None and np.isfinite(float(age)) else 0.0,
        sexe=_sexe(row),
        diagnostic=f"MDD — rTMS protocole {protocol} ({freq:g} Hz {site})",
    )

    params = _rtms_parameters(protocol)
    for e_idx, epoch in enumerate(epochs):
        session = SessionRTMS(
            id_session=f"{subject_id}-E{e_idx:02d}",
            patient_id=subject_id,
            parametres=params,
            date=REFERENCE_DATE,
        )
        for c_idx, canal in enumerate(channels):
            session.enregistrer_donnees(
                SignalNeurophysiologique(
                    type_signal=SignalType.EEG,
                    valeurs=np.ascontiguousarray(epoch[c_idx], dtype=np.float32),
                    timestamp=REFERENCE_DATE,
                    canal=canal,
                    sampling_rate_hz=fs,
                )
            )
        # Autonomic channel. Stored as the RR tachogram rather than the raw ECG
        # trace: HRV is what the model consumes, R-peak detection is expensive to
        # redo per prediction, and the tachogram is ~64 floats against 60 000.
        # sampling_rate_hz is 0.0 because an RR series is event-sampled, not
        # uniformly sampled — writing 250 Hz here would be a lie the app might plot.
        if tachogram is not None:
            session.enregistrer_donnees(
                SignalNeurophysiologique(
                    type_signal=SignalType.ECG,
                    valeurs=np.ascontiguousarray(tachogram[e_idx], dtype=np.float32),
                    timestamp=REFERENCE_DATE,
                    canal=ecg_channel,
                    sampling_rate_hz=0.0,
                )
            )
        # Scores are per treatment course, not per epoch — identical on each.
        session.score_pre = bdi_pre
        session.cloturer(score_post=bdi_post)
        patient.ajouter_session(session)

    patient.ajouter_dossier(
        DossierClinique(
            date=REFERENCE_DATE,
            note="BDI-II avant traitement rTMS (TDBRAIN).",
            score_depression=bdi_pre,
        )
    )
    patient.ajouter_dossier(
        DossierClinique(
            date=REFERENCE_DATE,
            note=(
                f"BDI-II après traitement — réduction {pct:.0%} → "
                f"{'répondeur' if responder else 'non-répondeur'} (seuil 50%)."
            ),
            score_depression=bdi_post,
        )
    )
    return patient


def seed_tdbrain(
    repo: Repository,
    dataset: LoadedDataset,
    limit: int | None = None,
    progress=None,
) -> int:
    """Write a TDBRAIN :class:`LoadedDataset` into the database.

    Requires the full montage (``dataset.signals_mc`` / ``dataset.channels``) —
    the model consumes 26-channel band powers, so seeding a single channel would
    make the stored data unusable for prediction.

    ``progress`` is an optional ``callable(done, total)`` for UI feedback.
    """
    if dataset.signals_mc is None or not dataset.channels:
        raise ValueError(
            "TDBRAIN seeding needs the full montage (signals_mc + channels); "
            "load with load_tdbrain(), not the simulated loader."
        )
    required = {"patient_id", "protocol", "bdi_pre", "bdi_post", "responder", "pct_reduction"}
    missing = required - set(dataset.metadata.columns)
    if missing:
        raise ValueError(
            f"metadata is missing {sorted(missing)} — seed from a task='response' "
            f"dataset (the diagnosis task carries no BDI/protocol columns)."
        )

    n = dataset.signals_mc.shape[0] if limit is None else min(limit, dataset.signals_mc.shape[0])
    channels = list(dataset.channels)
    count = 0
    for p_idx in range(n):
        patient = _patient_from_tdbrain(
            row=dataset.metadata.iloc[p_idx],
            epochs=dataset.signals_mc[p_idx],
            channels=channels,
            fs=dataset.fs,
            tachogram=None if dataset.ecg is None else dataset.ecg[p_idx],
        )
        repo.sauvegarder_patient(patient)
        count += 1
        if progress is not None:
            progress(count, n)
    return count


def dataset_from_repository(
    repo: Repository,
    patient_ids: list[str] | None = None,
    progress=None,
) -> LoadedDataset:
    """Inverse of :func:`seed_tdbrain` — rebuild a full dataset from the DB.

    Lets the app retrain on the cohort it already holds instead of re-reading the
    15 GB BDF archive. Returning a :class:`LoadedDataset` rather than bare arrays
    is what lets the app call :func:`src.data.modalities.build_features` — the
    *same* function training uses. Rebuilding features a second way here is how
    the app would start training a variant on a vector its checkpoint never saw.

    The metadata carries the clinical block's four columns (``protocol``, ``age``,
    ``gender``, ``bdi_pre``); the responder label is *recomputed* from the stored
    BDI-II scores rather than trusted from a cached column, so it always agrees
    with what the UI displays. Channels are ordered canonically
    (:data:`TDBRAIN_CHANNELS_26`), never by the database's row order.
    """
    from ..db.schema import PatientRow
    from .tdbrain import TDBRAIN_CHANNELS_26

    if patient_ids is None:
        from sqlalchemy import select

        with repo._Session() as s:
            patient_ids = list(
                s.execute(select(PatientRow.id).order_by(PatientRow.id)).scalars()
            )

    mc_list: list[np.ndarray] = []
    rr_list: list[np.ndarray | None] = []
    labels: list[int] = []
    groups: list[str] = []
    meta_rows: list[dict] = []
    channels: list[str] | None = None
    fs: float | None = None

    for i, pid in enumerate(patient_ids, start=1):
        patient = repo.charger_patient(pid)
        if patient is None or not patient.sessions:
            continue
        # Split by modality: the ECG tachogram shares the session but is neither a
        # montage channel nor uniformly sampled, so it must not enter `per_epoch`.
        per_epoch = [
            {sig.canal: sig.valeurs for sig in sess.signaux if sig.type_signal == SignalType.EEG}
            for sess in patient.sessions
        ]
        rr_epochs = [
            [sig.valeurs for sig in sess.signaux if sig.type_signal == SignalType.ECG]
            for sess in patient.sessions
        ]
        if not all(per_epoch):
            continue
        available = set.intersection(*(set(d) for d in per_epoch))
        order = [c for c in TDBRAIN_CHANNELS_26 if c in available]
        if not order:
            continue
        if channels is None:
            channels = order
            # Take fs from an EEG signal specifically: the ECG row stores 0.0.
            fs = float(
                next(s for s in patient.sessions[0].signaux if s.type_signal == SignalType.EEG)
                .sampling_rate_hz
            )
        elif order != channels:
            continue  # inconsistent montage — skip rather than misalign columns

        width = min(len(d[c]) for d in per_epoch for c in channels)
        mc_list.append(
            np.stack([np.stack([d[c][:width] for c in channels]) for d in per_epoch]).astype(np.float32)
        )
        rr_list.append(
            np.stack([e[0] for e in rr_epochs]).astype(np.float32)
            if all(len(e) == 1 for e in rr_epochs) else None
        )

        first = patient.sessions[0]
        pre, post = first.score_pre, first.score_post
        if pre is None or post is None or pre <= 0:
            mc_list.pop()
            rr_list.pop()
            continue
        pct = (pre - post) / pre
        labels.append(int(pct >= 0.5))
        groups.append(pid)
        meta_rows.append({
            "patient_id": pid,
            "protocol": protocol_from_parameters(first.parametres),
            "age": float(patient.age),
            # None (never seeded, or entered by hand) stays NaN so that
            # `modalities._impute` fills it with the cohort median and warns,
            # exactly as it does when participants.tsv has a gap.
            "gender": np.nan if patient.sexe is None else float(patient.sexe),
            "bdi_pre": float(pre),
            "bdi_post": float(post),
            "pct_reduction": float(pct),
            "responder": int(pct >= 0.5),
        })
        if progress is not None:
            progress(i, len(patient_ids))

    if not mc_list:
        raise ValueError(
            "no usable patients in the database — seed it first "
            "(python -m src.data.tdbrain_seeder --root <TDBRAIN root>)."
        )
    width = min(a.shape[-1] for a in mc_list)
    n_ep = min(a.shape[0] for a in mc_list)
    mc = np.stack([a[:n_ep, :, :width] for a in mc_list])

    # All-or-nothing on the autonomic block: a cohort where only some patients have
    # a tachogram cannot form a rectangular tensor, and zero-filling the rest would
    # invent a flat heart rate for them.
    ecg = None
    if rr_list and all(r is not None for r in rr_list):
        n_rr = min(r.shape[-1] for r in rr_list)
        ecg = np.stack([r[:n_ep, :n_rr] for r in rr_list]).astype(np.float32)

    return LoadedDataset(
        # A representative single channel, for the code paths that predate the
        # montage; `signals_mc` is what the multimodal models actually read.
        signals=mc[:, :, 0, :],
        labels=np.asarray(labels, dtype=np.int8),
        fs=float(fs),
        window=int(width),
        metadata=pd.DataFrame(meta_rows),
        ecg=ecg,
        channels=channels,
        signals_mc=mc,
    )


def montage_from_repository(
    repo: Repository,
    patient_ids: list[str] | None = None,
    progress=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float, np.ndarray | None]:
    """``(mc, labels, groups, channels, fs, ecg)`` — the array view of the DB.

    Kept as the narrow interface for callers that want the montage alone; both it
    and the app read the same :func:`dataset_from_repository` underneath.
    """
    ds = dataset_from_repository(repo, patient_ids, progress)
    return (
        ds.signals_mc,
        ds.labels,
        ds.metadata["patient_id"].to_numpy(),
        list(ds.channels),
        ds.fs,
        ds.ecg,
    )


def _main() -> None:
    import argparse
    import warnings

    from .tdbrain import TDBRAINConfig, load_tdbrain

    ap = argparse.ArgumentParser(description="Seed SQLite from a montage cohort.")
    ap.add_argument("--root", type=Path, default=None,
                    help="TDBRAIN root (the folder holding participants.tsv); "
                         "not needed with --matched")
    ap.add_argument("--matched", action="store_true",
                    help="seed the *matched simulated* cohort instead of the real "
                         "one — same montage shape, calibrated on TDBRAIN")
    ap.add_argument("--effect", type=float, default=0.0,
                    help="matched simulator effect size (0 reproduces the real null)")
    ap.add_argument("--seed", type=int, default=42, help="matched simulator seed")
    # Resolved per mode below: defaulting to recherche.sqlite3 would let a
    # TDBRAIN seed silently overwrite the sequential simulated cohort.
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--metadata", type=Path, default=None,
                    help="participants table (default: <root>/participants.tsv)")
    ap.add_argument("--col-id", default="TDBRAIN_ID")
    ap.add_argument("--col-protocol", default="rTMS PROTOCOL")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.matched:
        from .simulator_matched import MatchedSimConfig, simulate_matched

        db = args.db or Path("recherche_sim_matched.sqlite3")
        # The defaults must stay in step with src.models.train_all, or the app
        # would predict on a different cohort than the checkpoints were fit on.
        print(f"generating the matched simulated cohort (effect={args.effect})…")
        dataset = simulate_matched(
            MatchedSimConfig(effect_size=args.effect, seed=args.seed)
        )
        kind = "matched simulated"
    else:
        if args.root is None:
            ap.error("--root is required unless --matched is given")
        db = args.db or Path("recherche_tdbrain.sqlite3")
        cfg = TDBRAINConfig(
            root=args.root,
            metadata_path=args.metadata,
            col_id=args.col_id,
            col_protocol=args.col_protocol,
        )
        print(f"loading TDBRAIN from {args.root} (reads one BDF per patient)…")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = load_tdbrain(cfg)
        kind = "real TDBRAIN"
    print(f"loaded {dataset.signals_mc.shape[0]} patients, montage {dataset.signals_mc.shape}")

    repo = Repository(db_url=f"sqlite:///{db}")
    n = seed_tdbrain(repo, dataset, limit=args.limit,
                     progress=lambda d, t: print(f"  seeded {d}/{t}", end="\r"))
    print(f"\nSeeded {n} {kind} patients into {db}")


if __name__ == "__main__":
    _main()
