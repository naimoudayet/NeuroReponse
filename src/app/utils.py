from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import streamlit as st

from ..db import Repository
from ..models.variants import (
    Dataset,
    Task,
    Variant,
    VariantConfig,
    article_variants,
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
    # What the head predicts, and which rTMS arm it was fitted on. The article-
    # aligned variants (Arteaga et al.) are per-protocol regressions; the original
    # 2x2 is pooled classification, which is what these defaults describe.
    target: str = "responder"
    protocol: int | None = None
    task: Task = Task.CLASSIFICATION
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

    @property
    def is_regression(self) -> bool:
        return self.task is Task.REGRESSION


@dataclass(frozen=True)
class SourceConfig:
    """A cohort: one database, one or more models trained on it."""

    label: str
    db: Path
    caption: str
    models: tuple[ModelChoice, ...]
    # The article-aligned regression arm, kept in its own field so `models` still
    # means exactly "the 2x2's classification checkpoints" for existing callers.
    regression_models: tuple[ModelChoice, ...] = ()
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

    @property
    def all_models(self) -> tuple[ModelChoice, ...]:
        return self.models + self.regression_models

    @property
    def supports_regression(self) -> bool:
        return bool(self.regression_models)


def _from_variant(cfg: VariantConfig) -> ModelChoice:
    """Present a 2x2 variant as an app-level choice.

    The registry in ``src.models.variants`` stays the definition of the
    experiment; this only decides how it is *labelled* on screen.
    """
    return ModelChoice(
        key=cfg.key.value,
        # The article-aligned variants carry their own labels, because three
        # feature sets share this axis there ("EEG seul" vs "Clinique seul" vs
        # "Multimodal") and `uses_signals` cannot tell the first from the third.
        label=(
            cfg.label
            if cfg.task is Task.REGRESSION
            else (
                "Multimodal — clinique + EEG + ECG (139)"
                if cfg.uses_signals
                else "Clinique seul (4)"
            )
        ),
        model=cfg.model,
        caption=cfg.caption,
        uses_signals=cfg.uses_signals,
        modalities=tuple(cfg.modalities),
        variant=cfg.key,
        target=cfg.target,
        protocol=cfg.protocol,
        task=cfg.task,
    )


_SEQ_CAPTION = (
    "Cohorte générée par `src/data/simulator.py` — 10 **séances** rTMS par "
    "patient. C'est la seule source où le LSTM accumule l'information au fil du "
    "traitement ; elle ne fait pas partie de la comparaison 2×2."
)

# Cohorts offered in the sidebar. All three are selectable: the sequential cohort
# (the only multi-session treatment course, which the clinical loop needs), the
# matched simulated cohort and the real one. The last two are the two halves of
# the 2x2 comparison — the matched cohort was previously hidden on the grounds
# that it had "neither role", which stopped being true once the app had to serve
# the four comparison models rather than only describe them on the results page.
VISIBLE: tuple[DataSource, ...] = (
    DataSource.SIMULE_SEQ, DataSource.SIMULE, DataSource.TDBRAIN,
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
        regression_models=tuple(
            _from_variant(c) for c in article_variants(Dataset.SIMULE)
        ),
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
        regression_models=tuple(
            _from_variant(c) for c in article_variants(Dataset.TDBRAIN)
        ),
        unit="époque",
        is_real=True,
    ),
}

_STATE_KEY = "data_source"
_STATE_KEY_WIDGET = "data_source_widget"
_MODEL_STATE_KEY = "model_choice"
_MODEL_STATE_WIDGET = "model_choice_widget"
_PROTOCOL_STATE_KEY = "protocole"
_PROTOCOL_STATE_WIDGET = "protocole_widget"
_TASK_STATE_KEY = "objectif"
_TASK_STATE_WIDGET = "objectif_widget"

# The two rTMS arms are different treatments, not a nuisance variable: 10 Hz
# excitatory over the left DLPFC versus 1 Hz inhibitory over the right. The
# reference study (PMC12981298) fits one model per arm, so the app has to be able
# to select one — and, crucially, to show only that arm's patients alongside it.
# No pooled entry: the reference study fits one model per arm, so there is no
# pooled checkpoint to offer, and a cohort-wide model would answer a different
# question than the one the sidebar would be labelling.
PROTOCOLES: dict[int | None, str] = {
    1: "Protocole 1 — 10 Hz L-DLPFC",
    2: "Protocole 2 — 1 Hz R-DLPFC",
}

# What the model predicts. A cohort may hold checkpoints for both heads — the
# real cohort does — so this is a user choice, not a property of the cohort.
HEADS: dict[Task, str] = {
    Task.CLASSIFICATION: "Réponse binaire (répondeur / non-répondeur)",
    Task.REGRESSION: "Réduction du BDI-II (ΔBDI, continu)",
}

HEAD_CAPTIONS: dict[Task, str] = {
    Task.CLASSIFICATION: (
        "Comparaison 2×2 : deux cohortes × deux jeux de variables, cible binaire, "
        "**les deux bras rTMS regroupés**. C'est pourquoi aucun protocole n'est "
        "proposé ici — les points de contrôle correspondants sont mis en commun."
    ),
    Task.REGRESSION: (
        "Arm aligné sur l'étude de référence (Arteaga et al.) : cible continue, "
        "**un modèle par bras rTMS**, score = corrélation de Pearson."
    ),
}

# `None` remains a meaningful protocol value internally — the sequential cohort's
# model is not per-arm — so it cannot double as "argument not supplied".
_UNSET: object = object()

# Backwards-compatible defaults (legacy sequential track).
DB_PATH = SOURCES[DataSource.SIMULE_SEQ].db
MODEL_PATH = SOURCES[DataSource.SIMULE_SEQ].model
SIM_DIR = Path("data/simulated")


def current_source() -> DataSource:
    return DataSource(st.session_state.get(_STATE_KEY, DataSource.SIMULE_SEQ.value))


def source_config(source: DataSource | None = None) -> SourceConfig:
    return SOURCES[source or current_source()]


def available_heads(cfg: SourceConfig) -> tuple[Task, ...]:
    """Heads this cohort can actually serve, most-default first.

    Driven by **what is on disk**, not by what the registry declares: a head
    whose checkpoints were never trained must never appear in the sidebar, which
    was the original and still-valid reason this was not a user choice. What
    changed is that it is no longer a per-cohort constant — the real cohort holds
    both the 2x2's pooled classification pair *and* the article arm's per-protocol
    regressions, so exactly one of them was unreachable.

    Regression sorts first whenever the cohort has one, because that was the
    derived value before the head became selectable; every previously recorded
    default therefore still resolves to the same checkpoint.
    """
    present = {m.task for m in cfg.all_models if m.model.exists()}
    order = (
        (Task.REGRESSION, Task.CLASSIFICATION)
        if cfg.supports_regression
        else (Task.CLASSIFICATION, Task.REGRESSION)
    )
    return tuple(t for t in order if t in present)


def current_task(source: DataSource | None = None) -> Task:
    """The selected head, falling back to this cohort's default.

    A stored value naming a head the cohort cannot serve — a session resumed
    after the checkpoints changed, or a cohort switch — falls back rather than
    raising, the same rule :func:`source_selector` applies to a hidden cohort.
    """
    cfg = source_config(source)
    heads = available_heads(cfg)
    if not heads:
        # No checkpoint on disk at all: keep the old derived answer so the pages
        # can still explain themselves instead of failing to render.
        return Task.REGRESSION if cfg.supports_regression else Task.CLASSIFICATION
    raw = st.session_state.get(f"{_TASK_STATE_KEY}:{cfg.label}")
    for head in heads:
        if head.value == raw:
            return head
    return heads[0]


def current_protocol(source: DataSource | None = None) -> int | None:
    """Selected rTMS arm, or ``None`` for pooled.

    Only the regression arm is fitted per protocol; the original 2x2 pooled both,
    so asking for an arm there would promise a filter no checkpoint honours.

    **The head decides, not just the cohort.** The 2x2's classification
    checkpoints carry ``protocol=None``. Returning an arm while that head is
    selected would make :func:`available_models` filter on ``m.protocol == 1``,
    match nothing, and let :func:`model_choice` fall back to ``cfg.models[0]`` —
    serving a checkpoint the sidebar never selected, with the patient list
    filtered to one arm and nothing on screen looking wrong. That is the same
    silent failure the ``_UNSET`` sentinel exists to prevent.
    """
    cfg = source_config(source)
    if not cfg.supports_regression:
        return None
    if current_task(source) is Task.CLASSIFICATION:
        return None
    raw = st.session_state.get(f"{_PROTOCOL_STATE_KEY}:{cfg.label}")
    # Protocol 1 by default: every article-aligned checkpoint is fitted on one
    # arm, so there is no pooled model to fall back to.
    return raw if raw in (1, 2) else 1


def available_models(
    source: DataSource | None = None,
    task: Task | None = None,
    protocol: int | None | object = _UNSET,
) -> tuple[ModelChoice, ...]:
    """Checkpoints matching the selected axes, in registry order.

    A protocol-specific request must never fall back to a pooled checkpoint: the
    pooled model was fitted across **both** arms, so serving it under a
    "Protocole 1" heading answers a different question than the one on screen —
    while the page, having filtered its patient list to protocol 1, looks
    entirely consistent.

    ``protocol`` defaults to the *selected* arm, not to ``None``. Defaulting it to
    ``None`` made every caller that omitted it — which is every page, via
    :func:`model_choice` — silently receive the pooled checkpoint while the
    sidebar said "Protocole 1".
    """
    cfg = source_config(source)
    task = task or current_task(source)
    arm = current_protocol(source) if protocol is _UNSET else protocol
    return tuple(
        m for m in cfg.all_models
        if m.task is task and m.protocol == arm
    )


def model_choice(source: DataSource | None = None) -> ModelChoice:
    """The model selected for this cohort, defaulting to its first."""
    cfg = source_config(source)
    options = available_models(source) or cfg.models
    key = st.session_state.get(f"{_MODEL_STATE_KEY}:{cfg.label}")
    for choice in options:
        if choice.key == key:
            return choice
    return options[0]


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
    options = list(VISIBLE)
    current = current_source()
    # A stored selection can name a cohort that is no longer offered (the matched
    # simulated one, or a session resumed after the roster changed). Fall back to
    # the first visible cohort rather than letting `.index()` raise on every page.
    if current not in options:
        current = options[0]
        st.session_state[_STATE_KEY] = current.value
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

    # --- objectif: which head, when the cohort holds checkpoints for both ----- #
    # Hidden (not disabled) when a cohort serves a single head, for the same
    # reason the feature-set radio is: a control with one option is a claim that
    # a choice exists.
    heads = available_heads(cfg)
    picked_task = current_task(chosen)
    if len(heads) > 1:
        picked_task = container.radio(
            "Objectif",
            heads,
            index=heads.index(picked_task) if picked_task in heads else 0,
            format_func=lambda t: HEADS[t],
            key=f"{_TASK_STATE_WIDGET}:{cfg.label}",
        )
        st.session_state[f"{_TASK_STATE_KEY}:{cfg.label}"] = picked_task.value
        container.caption(HEAD_CAPTIONS[picked_task])

    # --- rTMS arm: every article-aligned checkpoint is fitted on one arm ------ #
    # Regression only. The 2x2's classification checkpoints pool both arms, so an
    # arm radio there would offer a filter no checkpoint honours.
    if cfg.supports_regression and picked_task is Task.REGRESSION:
        arms = [a for a in PROTOCOLES if available_models(chosen, picked_task, a)]
        current_arm = current_protocol(chosen)
        picked_arm = container.radio(
            "Protocole rTMS",
            arms,
            index=arms.index(current_arm) if current_arm in arms else 0,
            format_func=lambda a: PROTOCOLES[a],
            key=f"{_PROTOCOL_STATE_WIDGET}:{cfg.label}",
        )
        st.session_state[f"{_PROTOCOL_STATE_KEY}:{cfg.label}"] = picked_arm
        container.caption(
            "La liste des patients est filtrée sur ce bras : les deux protocoles "
            "sont des traitements différents, et l'étude de référence les modélise "
            "séparément."
        )
    else:
        picked_arm = None

    # --- feature set --------------------------------------------------------- #
    options = available_models(chosen, picked_task, picked_arm) or cfg.models
    if len(options) > 1:
        keys = [m.key for m in options]
        active = model_choice(chosen)
        picked = container.radio(
            "Jeu de variables",
            keys,
            index=keys.index(active.key) if active.key in keys else 0,
            format_func=lambda k: next(m.label for m in options if m.key == k),
            key=f"{_MODEL_STATE_WIDGET}:{cfg.label}",
        )
        st.session_state[f"{_MODEL_STATE_KEY}:{cfg.label}"] = picked
        container.caption(next(m.caption for m in options if m.key == picked))
    elif options:
        st.session_state[f"{_MODEL_STATE_KEY}:{cfg.label}"] = options[0].key
        container.caption(options[0].caption)
    return chosen


@st.cache_resource
def _repository_for(db_url: str) -> Repository:
    return Repository(db_url=db_url)


def get_repository(source: DataSource | None = None) -> Repository:
    return _repository_for(f"sqlite:///{source_config(source).db}")


def patient_protocols(repo: Repository) -> dict[str, int | None]:
    """``{patient_id: rTMS protocol}``, derived from each patient's stimulation
    parameters — the same mapping training used to split the cohort.

    Read from the stored parameters rather than a column, because the protocol is
    not persisted as one: it is recovered from frequency + site by
    ``protocol_from_parameters``, which is the single definition both the seeder
    and the clinical block already use.
    """
    import json

    from sqlalchemy import select

    from ..data.tdbrain_seeder import protocol_from_parameters
    from ..db.schema import SessionRow
    from ..domain import RTMSParameters

    out: dict[str, int | None] = {}
    with repo._Session() as s:
        rows = s.execute(
            select(SessionRow.patient_id, SessionRow.parametres_json)
            .order_by(SessionRow.patient_id, SessionRow.id_session)
        ).all()
    for patient_id, params_json in rows:
        if patient_id in out:
            continue                      # first session decides; they all agree
        out[patient_id] = protocol_from_parameters(
            RTMSParameters(**json.loads(params_json))
        )
    return out


def list_patient_ids(
    repo: Repository, protocol: int | None = None
) -> list[str]:
    """Patient ids, optionally restricted to one rTMS arm.

    The filter is not cosmetic. A checkpoint fitted on protocol 1 offered a
    protocol-2 patient would predict across treatment arms and report it as a
    normal result — the page has no way to notice, because the feature vector is
    the right shape either way.
    """
    from sqlalchemy import select

    from ..db.schema import PatientRow

    with repo._Session() as s:
        ids = list(s.execute(select(PatientRow.id).order_by(PatientRow.id)).scalars())
    if protocol is None:
        return ids
    by_patient = patient_protocols(repo)
    return [pid for pid in ids if by_patient.get(pid) == protocol]


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
