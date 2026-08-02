"""Rebuild a stored patient's model inputs — shared by every page that predicts.

Both the Predictions page and the follow-up (Suivi) page need the exact same
tensor, and building it twice invites drift: a divergence in channel order,
normalisation or modality set would make the two pages disagree about the same
patient while both look correct. So the reconstruction lives here once.

The TDBRAIN path is contract-driven: channels are selected **by name** in the
sidecar's order (never by database row order) and the HRV block is appended only
when the contract declares the ECG modality — with the same absence of
normalisation used at training time.

**Blocks are concatenated in the canonical order ``rtms, eeg, ecg``**, matching
:data:`src.data.modalities.MODALITY_ORDER` rather than the order the contract
happens to list them in. Training assembles them that way, so anything else here
would feed a permuted vector to a model that never saw one.

The clinical block round-trips **exactly**: protocol is recovered from the stored
stimulation parameters, and age is stored unrounded precisely because it is the
strongest single predictor in this project — an age rounded on the way into SQLite
would make the app predict on a value the model never saw.
"""
from __future__ import annotations

import numpy as np

from ..domain import SignalType


def clinical_block(patient, n_epochs: int) -> np.ndarray:
    """``(n_epochs, 4)`` clinical features, in ``RTMS_FEATURE_NAMES`` order.

    Repeated along the epoch axis exactly as :func:`src.data.modalities.rtms_block`
    repeats it at training time: these describe the patient and the treatment
    course, not the recording window.

    A missing covariate raises rather than imputing. Training imputes with the
    *cohort* median, which a single patient at prediction time cannot compute —
    substituting anything else here would quietly feed the model a different
    number than the one it was fit on.
    """
    from ..data.modalities import RTMS_FEATURE_NAMES
    from ..data.tdbrain_seeder import protocol_from_parameters

    if not patient.sessions:
        raise ValueError(
            "le bloc clinique exige au moins une séance (paramètres de stimulation)"
        )

    protocol = protocol_from_parameters(patient.sessions[0].parametres)
    if protocol is None:
        raise ValueError(
            "protocole rTMS non reconnu : ni 10 Hz L-DLPFC ni 1 Hz R-DLPFC"
        )

    bdi_pre = patient.sessions[0].score_pre
    if bdi_pre is None:
        for dossier in patient.historique_clinique:
            if dossier.score_depression is not None:
                bdi_pre = dossier.score_depression
                break
    if bdi_pre is None:
        raise ValueError("BDI-II de référence absent (score_pre / historique)")
    if patient.sexe is None:
        raise ValueError(
            "sexe non renseigné — le modèle clinique attend cette variable "
            "(0/1, comme dans la table source)"
        )

    values = {
        "rtms_protocol": float(protocol),
        "age": float(patient.age),
        "gender": float(patient.sexe),
        "bdi_pre": float(bdi_pre),
    }
    row = np.asarray([values[name] for name in RTMS_FEATURE_NAMES], dtype=np.float32)
    return np.tile(row, (n_epochs, 1))


def rebuild_tdbrain_input(patient, contract, fs: float) -> np.ndarray:
    """``(1, n_epochs, input_size)`` for a stored patient, per its contract.

    Raises ``ValueError``/``KeyError`` when the stored data does not match the
    contract, so callers can surface the reason instead of predicting on a
    silently mis-shaped vector.
    """
    from ..data.tdbrain import montage_band_powers, zscore_epochs
    from ..preprocessing.features import hrv_features
    from ..preprocessing.pipeline import HRV_FEATURE_NAMES

    # A clinical-only model reads nothing from the recordings, so a patient with
    # no signals at all (one created by hand in the app) is still predictable.
    if not contract.uses_signals:
        x = clinical_block(patient, contract.n_epochs)
        return _checked(x, contract)

    per_epoch = [
        {s.canal: s.valeurs for s in sess.signaux if s.type_signal == SignalType.EEG}
        for sess in patient.sessions
    ]
    tachograms = [
        [s.valeurs for s in sess.signaux if s.type_signal == SignalType.ECG]
        for sess in patient.sessions
    ]
    has_ecg = bool(tachograms) and all(len(t) == 1 for t in tachograms)

    available = set.intersection(*(set(d) for d in per_epoch))
    ok, why = contract.matches(available, fs, len(per_epoch), has_ecg=has_ecg)
    if not ok:
        raise ValueError(why)

    blocks: list[np.ndarray] = []
    if "rtms" in contract.modalities:
        blocks.append(clinical_block(patient, len(per_epoch)))

    if "eeg" in contract.modalities:
        width = min(len(d[c]) for d in per_epoch for c in contract.channels)
        mc = np.stack(
            [np.stack([d[c][:width] for c in contract.channels]) for d in per_epoch]
        ).astype(np.float32)
        eeg = montage_band_powers(mc, fs)
        if contract.per_patient_zscore:
            eeg = zscore_epochs(eeg)
        blocks.append(eeg)

    if "ecg" in contract.modalities:
        blocks.append(np.stack([
            [hrv_features(t[0])[n] for n in HRV_FEATURE_NAMES] for t in tachograms
        ]).astype(np.float32))

    return _checked(np.concatenate(blocks, axis=-1), contract)


def _checked(x: np.ndarray, contract) -> np.ndarray:
    """Refuse a vector the model cannot eat, and add the batch axis."""
    if x.shape[-1] != contract.input_size:
        raise ValueError(
            f"vecteur reconstruit de {x.shape[-1]} dimensions, "
            f"le modèle en attend {contract.input_size}"
        )
    return x[np.newaxis, :, :]


def snapshot_input(session, contract, patient=None) -> np.ndarray:
    """``(1, n_epochs, input_size)`` from **one** session holding a full recording.

    This is the TDBRAIN model's legitimate use inside a clinical loop. That model
    was trained on ``n_epochs`` windows of a *single* resting recording, so each
    loop iteration must re-record and predict on that recording alone — the
    accumulation happens at the clinical level (a trend of independent
    predictions), never by feeding sessions weeks apart into the recurrent axis,
    which is a distribution the model has never seen.

    The session therefore stores each montage channel as the **whole** recording;
    the epochs are cut here, exactly as ``load_tdbrain`` cuts them, so the tensor
    is byte-for-byte the shape the contract was written against.
    """
    from ..data.tdbrain import montage_band_powers, zscore_epochs
    from ..preprocessing.features import hrv_features
    from ..preprocessing.pipeline import HRV_FEATURE_NAMES

    by_name = {
        s.canal: s.valeurs for s in session.signaux if s.type_signal == SignalType.EEG
    }
    missing = [c for c in contract.channels if c not in by_name]
    if missing:
        raise ValueError(
            f"canaux absents de l'enregistrement : {missing[:5]}"
            f"{'…' if len(missing) > 5 else ''} "
            f"(le modèle en attend {len(contract.channels)})"
        )

    need = contract.window * contract.n_epochs
    short = [c for c in contract.channels if len(by_name[c]) < need]
    if short:
        raise ValueError(
            f"enregistrement trop court : {len(by_name[short[0]])} échantillons pour "
            f"{short[0]}, il en faut {need} "
            f"({need / contract.fs:.0f} s à {contract.fs:g} Hz)"
        )

    # (n_channels, n_epochs, window) -> (n_epochs, n_channels, window)
    mc = np.stack([
        np.asarray(by_name[c][:need], dtype=np.float32).reshape(
            contract.n_epochs, contract.window)
        for c in contract.channels
    ]).transpose(1, 0, 2)

    blocks: list[np.ndarray] = []
    if "rtms" in contract.modalities:
        if patient is None:
            raise ValueError(
                "ce modèle utilise le bloc clinique : le patient doit être fourni "
                "en plus de l'enregistrement (âge, sexe, BDI-II, protocole)"
            )
        blocks.append(clinical_block(patient, contract.n_epochs))

    eeg = montage_band_powers(mc, float(contract.fs))
    if contract.per_patient_zscore:
        eeg = zscore_epochs(eeg)
    blocks.append(eeg)

    if "ecg" in contract.modalities:
        tach = [s.valeurs for s in session.signaux if s.type_signal == SignalType.ECG]
        if len(tach) != 1:
            raise ValueError(
                "le modèle attend la modalité ECG mais l'enregistrement ne contient "
                "pas de tachogramme RR"
            )
        # HRV is measured once over the whole recording, so it repeats across the
        # epoch axis — identical to how load_tdbrain builds it.
        feats = hrv_features(tach[0])
        row = np.asarray([feats[n] for n in HRV_FEATURE_NAMES], dtype=np.float32)
        blocks.append(np.tile(row, (eeg.shape[0], 1)))

    return _checked(np.concatenate(blocks, axis=-1), contract)


def rebuild_simulated_input(patient, fs: float) -> np.ndarray:
    """``(1, n_sessions, n_features)`` for a simulated patient (single channel)."""
    from ..preprocessing.pipeline import PipelineConfig, preprocess

    window = patient.sessions[0].signaux[0].valeurs.shape[0]
    signals = np.stack([
        sess.signaux[0].valeurs[:window] for sess in patient.sessions
    ]).astype(np.float32)
    return preprocess(signals[np.newaxis, :, :], PipelineConfig(fs=fs, mode="features")).x


def eeg_sampling_rate(patient) -> float:
    """Sampling rate of the patient's EEG, ignoring the event-sampled ECG row."""
    sigs = patient.sessions[0].signaux
    eeg = [s for s in sigs if s.type_signal == SignalType.EEG]
    return float((eeg[0] if eeg else sigs[0]).sampling_rate_hz)


def build_model_input(patient, is_real: bool, contract=None) -> tuple[np.ndarray, float]:
    """Dispatch to the right reconstruction. Returns ``(x, fs)``.

    The **contract** decides, not ``is_real``: the four comparison variants each
    ship one, and a clinical-only variant must not be sent down the montage path
    just because its cohort is real. ``is_real`` survives only to catch a real
    cohort whose checkpoint lost its sidecar, which is never safe to guess at.
    """
    if contract is None:
        if is_real:
            raise ValueError("un contrat de features est requis pour la source TDBRAIN")
        fs = eeg_sampling_rate(patient)
        return rebuild_simulated_input(patient, fs), fs

    # A clinical-only model reads no recording, so it must not require one to
    # exist just to report a sampling rate.
    fs = eeg_sampling_rate(patient) if contract.uses_signals else float(contract.fs)
    return rebuild_tdbrain_input(patient, contract, fs), fs
