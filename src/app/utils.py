from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import streamlit as st

from ..db import Repository
from ..models.variants import (
    Dataset,
    Variant,
    VariantConfig,
    variant_config,
    variants_for,
)


class DataSource(str, Enum):
    """Which cohort the app is working against.

    Three, not two. The cohorts are kept in **separate SQLite files** on purpose:
    simulated and real patients must never be mixed in one table, and each model
    is trained on its own feature contract.

    ``SIMULE_SEQ`` and ``SIMULE`` are both simulated but they are *not* the same
    cohort. The first is the original 10-session treatment trajectory, the only
    place in this project where the LSTM accumulates evidence across sessions.
    The second is the TDBRAIN-matched cohort the 2x2 comparison is built on:
    baseline-only, 8 epochs of one resting recording. Collapsing them would
    either destroy the sequential demonstration or feed the comparison models a
    cohort they were never fit on.
    """

    SIMULE_SEQ = "simule_seq"
    SIMULE = "simule"
    TDBRAIN = "tdbrain"


@dataclass(frozen=True)
class ModelChoice:
    """One trained model offered for a cohort — the feature-set axis."""

    key: str
    label: str
    model: Path
    caption: str
    uses_signals: bool
    # Which feature blocks this model eats, in canonical order. Empty for the
    # legacy checkpoint, which predates the modality system and is rebuilt by the
    # fixed 8-feature pipeline instead.
    modalities: tuple[str, ...] = ()
    # The 2x2 registry entry this came from, when it is one of the four.
    variant: Variant | None = None
    # Every model trained by `train_all` ships a JSON feature contract; the
    # legacy sequential checkpoint predates them and rebuilds its inputs from the
    # fixed 8-feature pipeline instead. Anything else missing a sidecar is a bug,
    # not a fallback, so the pages refuse rather than guess.
    requires_contract: bool = True

    @property
    def sidecar(self) -> Path:
        return self.model.with_suffix(".json")


@dataclass(frozen=True)
class SourceConfig:
    """A cohort: one database, one or more models trained on it."""

    label: str
    db: Path
    caption: str
    models: tuple[ModelChoice, ...]
    unit: str = "séance"
    is_real: bool = False
    # True when the sequence axis is a real treatment trajectory, so the LSTM
    # accumulates evidence across sessions. False when it is epochs of a single
    # baseline recording, where each session is an independent snapshot and a
    # "trend" would be clinical, not recurrent.
    sequentiel: bool = False

    @property
    def model(self) -> Path:
        """First checkpoint — kept so callers wanting *a* path still work."""
        return self.models[0].model


def _from_variant(cfg: VariantConfig) -> ModelChoice:
    """Present a 2x2 variant as an app-level choice.

    The registry in ``src.models.variants`` stays the definition of the
    experiment; this only decides how it is *labelled* on screen.
    """
    return ModelChoice(
        key=cfg.key.value,
        label=(
            "Multimodal — clinique + EEG + ECG (139)"
            if cfg.uses_signals
            else "Clinique seul (4)"
        ),
        model=cfg.model,
        caption=cfg.caption,
        uses_signals=cfg.uses_signals,
        modalities=tuple(cfg.modalities),
        variant=cfg.key,
    )


_SEQ_CAPTION = (
    "Cohorte générée par `src/data/simulator.py` — 10 **séances** rTMS par "
    "patient. C'est la seule source où le LSTM accumule l'information au fil du "
    "traitement ; elle ne fait pas partie de la comparaison 2×2."
)

SOURCES: dict[DataSource, SourceConfig] = {
    DataSource.SIMULE_SEQ: SourceConfig(
        label="Simulé — séquentiel (10 séances)",
        db=Path("recherche.sqlite3"),
        caption=_SEQ_CAPTION,
        models=(
            ModelChoice(
                key="sim_seq",
                label="Séquentiel (8 variables EEG)",
                model=Path("data/models/lstm_v1.pt"),
                caption=_SEQ_CAPTION,
                uses_signals=True,
                requires_contract=False,
            ),
        ),
        sequentiel=True,
    ),
    DataSource.SIMULE: SourceConfig(
        label="Simulé — apparié TDBRAIN (8 époques)",
        db=variant_config(Variant.SIM_RTMS).db,
        caption=(
            "Cohorte simulée **calibrée sur TDBRAIN** (132 patients, 26 canaux + "
            "ECG). Même forme que la cohorte réelle : les « séances » sont des "
            "**époques** d'un enregistrement de repos unique."
        ),
        models=tuple(_from_variant(c) for c in variants_for(Dataset.SIMULE)),
        unit="époque",
    ),
    DataSource.TDBRAIN: SourceConfig(
        label="TDBRAIN (EEG réel)",
        db=variant_config(Variant.TDBRAIN_RTMS).db,
        caption=(
            "Cohorte réelle TDBRAIN (MDD traités par rTMS). Un seul enregistrement "
            "de repos par patient : les « séances » sont des **époques** de cet "
            "enregistrement, pas une évolution au fil du traitement."
        ),
        models=tuple(_from_variant(c) for c in variants_for(Dataset.TDBRAIN)),
        unit="époque",
        is_real=True,
    ),
}

_STATE_KEY = "data_source"
_STATE_KEY_WIDGET = "data_source_widget"
_MODEL_STATE_KEY = "model_choice"
_MODEL_STATE_WIDGET = "model_choice_widget"

# Backwards-compatible defaults (legacy sequential track).
DB_PATH = SOURCES[DataSource.SIMULE_SEQ].db
MODEL_PATH = SOURCES[DataSource.SIMULE_SEQ].model
SIM_DIR = Path("data/simulated")


def current_source() -> DataSource:
    return DataSource(st.session_state.get(_STATE_KEY, DataSource.SIMULE_SEQ.value))


def source_config(source: DataSource | None = None) -> SourceConfig:
    return SOURCES[source or current_source()]


def model_choice(source: DataSource | None = None) -> ModelChoice:
    """The model selected for this cohort, defaulting to its first."""
    cfg = source_config(source)
    key = st.session_state.get(f"{_MODEL_STATE_KEY}:{cfg.label}")
    for choice in cfg.models:
        if choice.key == key:
            return choice
    return cfg.models[0]


def source_selector(sidebar: bool = True) -> DataSource:
    """Render the cohort picker (and the feature-set picker) and return the source.

    Two axes, because the comparison is a 2x2: choosing a cohort alone would
    leave the app unable to show that 4 clinical variables beat 139 multimodal
    ones. The second radio is hidden when a cohort offers a single model, rather
    than shown disabled.

    Streamlit shares ``session_state`` across pages, so choosing here sticks for
    the whole app.
    """
    container = st.sidebar if sidebar else st
    options = list(SOURCES)
    current = current_source()
    chosen = container.radio(
        "Cohorte",
        options,
        index=options.index(current),
        format_func=lambda s: SOURCES[s].label,
        key=_STATE_KEY_WIDGET,
    )
    st.session_state[_STATE_KEY] = chosen.value
    container.caption(SOURCES[chosen].caption)

    cfg = SOURCES[chosen]
    if len(cfg.models) > 1:
        keys = [m.key for m in cfg.models]
        active = model_choice(chosen)
        picked = container.radio(
            "Jeu de variables",
            keys,
            index=keys.index(active.key),
            format_func=lambda k: next(m.label for m in cfg.models if m.key == k),
            key=f"{_MODEL_STATE_WIDGET}:{cfg.label}",
        )
        st.session_state[f"{_MODEL_STATE_KEY}:{cfg.label}"] = picked
        container.caption(next(m.caption for m in cfg.models if m.key == picked))
    return chosen


@st.cache_resource
def _repository_for(db_url: str) -> Repository:
    return Repository(db_url=db_url)


def get_repository(source: DataSource | None = None) -> Repository:
    return _repository_for(f"sqlite:///{source_config(source).db}")


def list_patient_ids(repo: Repository) -> list[str]:
    from sqlalchemy import select

    from ..db.schema import PatientRow

    with repo._Session() as s:
        return list(s.execute(select(PatientRow.id).order_by(PatientRow.id)).scalars())


def db_is_empty(repo: Repository) -> bool:
    return len(list_patient_ids(repo)) == 0


def seed_demo_data(
    repo: Repository, limit: int | None = None, source: DataSource | None = None
) -> int:
    """Seed the selected *simulated* cohort (generating it first if needed).

    The real cohort is not seedable from the app: it needs the gated ~14.5 GB
    TDBRAIN download and minutes of BDF decoding, so it stays a command-line
    operation rather than a button that appears to hang.
    """
    source = source or current_source()
    if source is DataSource.TDBRAIN:
        raise ValueError(
            "La cohorte TDBRAIN se charge en ligne de commande "
            "(`python -m src.data.tdbrain_seeder --root <TDBRAIN>`) : elle exige "
            "les données sous accès contrôlé et la lecture de 132 fichiers BDF."
        )

    if source is DataSource.SIMULE:
        from ..data.simulator_matched import MatchedSimConfig, simulate_matched
        from ..data.tdbrain_seeder import seed_tdbrain

        # Defaults must stay in step with src.models.train_all, or the app would
        # hold a different cohort than the checkpoints were fit on.
        return seed_tdbrain(
            repo, simulate_matched(MatchedSimConfig()), limit=limit
        )

    from ..data.loader import load
    from ..data.seeder import seed

    if not (SIM_DIR / "eeg_simulated.npz").exists():
        from ..data.simulator import save, simulate

        ds = simulate()
        save(ds, SIM_DIR)

    return seed(repo, load(SIM_DIR), limit=limit)


def model_path(source: DataSource | None = None) -> Path:
    return model_choice(source).model


def has_trained_model(source: DataSource | None = None) -> bool:
    return model_path(source).exists()


def format_rtms_parameters(params) -> dict:
    """Render rTMS parameters for display, hiding values the source never published.

    TDBRAIN publishes the protocol (frequency + site) but **not** the per-patient
    stimulation dose, which the seeder stores as ``0.0`` rather than inventing
    plausible numbers (see ``src/data/tdbrain_seeder.py``). Showing those raw would
    put "Intensité : 0 %", "Nb trains : 0" and "Total pulses : 0" on screen, which
    reads as *measured zero* rather than *not reported*. A zero here is never
    physically meaningful — you cannot stimulate at 0% or deliver 0 trains — so a
    dash plus the protocol note is both truer and clearer.
    """
    unknown = "— (non publié)"

    def val(x, suffix=""):
        return unknown if not x else f"{x:g}{suffix}"

    dose_known = all([
        params.intensite_pct, params.duree_train_s, params.nb_trains,
    ])
    return {
        "Fréquence (Hz)": val(params.frequence_hz),
        "Intensité (%)": val(params.intensite_pct),
        "Durée train (s)": val(params.duree_train_s),
        "Nb trains": val(params.nb_trains),
        "Intervalle train (s)": val(params.intervalle_train_s),
        "Localisation": params.localisation or unknown,
        "Protocole": params.protocole,
        "Total pulses": f"{params.total_pulses():g}" if dose_known else unknown,
    }
