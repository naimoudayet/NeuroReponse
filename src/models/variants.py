"""The four trained models, defined once.

Training (:mod:`src.models.train_all`) and the Streamlit app both need to know
which model exists, what it eats, where its checkpoint lives and which database
its patients come from. Defining that in two places is how a checkpoint ends up
being fed the wrong feature vector, so it is defined here and imported by both.

The four variants form a 2x2: two cohorts (simulated, real) crossed with two
feature sets (clinical only, clinical + neurophysiology). That crossing is the
experiment — it separates "does the signal help?" from "does the cohort matter?",
which neither axis alone can answer.

``ORDERED`` is the order the app must present them in.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

MODELS_DIR = Path("data/models")

# The *matched* simulated cohort, not the legacy sequential one: these two
# variants were fit on `simulator_matched.simulate_matched()` (132 patients, 8
# baseline epochs, 26 channels + ECG), which is a different cohort and a
# different shape from `simulator.py`'s 100 x 10 single-channel trajectory.
# Pointing this at recherche.sqlite3 would feed the checkpoints another cohort's
# patients. The legacy database is still used by the app, under its own entry.
SIM_DB = Path("recherche_sim_matched.sqlite3")
TDBRAIN_DB = Path("recherche_tdbrain.sqlite3")


class Dataset(str, Enum):
    SIMULE = "simule"
    TDBRAIN = "tdbrain"


class Task(str, Enum):
    """What the head predicts. See `src.models.lstm.LSTMConfig.task`."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class Variant(str, Enum):
    # The original 2x2: binary responder, both protocols pooled.
    SIM_RTMS = "sim_rtms"
    TDBRAIN_RTMS = "tdbrain_rtms"
    SIM_MULTI = "sim_multi"
    TDBRAIN_MULTI = "tdbrain_multi"

    # The article-aligned arm: continuous BDI-II change, protocols modelled
    # separately, mirroring Arteaga et al. (PMC12981298).
    #
    # Three feature sets per protocol, and each earns its place:
    #   *_EEG_REG   — 130 band powers, no clinical variables. This is the one that
    #                 matches the study, whose model sees **only** EEG.
    #   *_CLIN_REG  — the 4 clinical variables alone: the bar to clear, since
    #                 baseline BDI-II alone reaches r = 0.500 on protocol 1.
    #   *_MULTI_REG — both, plus HRV. This project's own multimodal question.
    #
    # Reading EEG-only against clinical-only is the comparison that says whether
    # the EEG contributed anything; the multimodal row cannot answer it, because
    # baseline BDI-II is one of its own inputs.
    TDBRAIN_P1_EEG_REG = "tdbrain_p1_eeg_reg"
    TDBRAIN_P2_EEG_REG = "tdbrain_p2_eeg_reg"
    TDBRAIN_P1_CLIN_REG = "tdbrain_p1_clin_reg"
    TDBRAIN_P2_CLIN_REG = "tdbrain_p2_clin_reg"
    TDBRAIN_P1_MULTI_REG = "tdbrain_p1_multi_reg"
    TDBRAIN_P2_MULTI_REG = "tdbrain_p2_multi_reg"


@dataclass(frozen=True)
class VariantConfig:
    key: Variant
    label: str                       # shown in the app's source selector
    dataset: Dataset
    modalities: tuple[str, ...]
    db: Path
    model: Path
    caption: str
    # Article-aligned axes. Defaulted so the original 2x2 entries are unchanged.
    task: Task = Task.CLASSIFICATION
    target: str = "responder"
    protocol: int | None = None          # None = both arms pooled

    @property
    def sidecar(self) -> Path:
        return self.model.with_suffix(".json")

    @property
    def is_regression(self) -> bool:
        return self.task is Task.REGRESSION

    @property
    def protocol_label(self) -> str:
        return {
            1: "Protocole 1 — 10 Hz L-DLPFC",
            2: "Protocole 2 — 1 Hz R-DLPFC",
        }.get(self.protocol, "Les deux protocoles")

    @property
    def uses_signals(self) -> bool:
        """Whether this variant needs recorded signals, not just the metadata."""
        return bool({"eeg", "ecg"} & set(self.modalities))


_CLINICAL_CAPTION = (
    "Modèle **clinique** : protocole rTMS, âge, sexe et BDI-II de référence. "
    "Aucun signal neurophysiologique — sert de référence à battre."
)
_MULTI_CAPTION = (
    "Modèle **multimodal** : paramètres cliniques + 130 puissances de bande EEG "
    "(26 canaux × 5 bandes) + 5 métriques HRV issues de l'ECG."
)

VARIANTS: dict[Variant, VariantConfig] = {
    Variant.SIM_RTMS: VariantConfig(
        key=Variant.SIM_RTMS,
        label="Données simulées — rTMS",
        dataset=Dataset.SIMULE,
        modalities=("rtms",),
        db=SIM_DB,
        model=MODELS_DIR / "sim_rtms_v1.pt",
        caption="Cohorte simulée, calibrée sur TDBRAIN. " + _CLINICAL_CAPTION,
    ),
    Variant.TDBRAIN_RTMS: VariantConfig(
        key=Variant.TDBRAIN_RTMS,
        label="TDBRAIN — rTMS",
        dataset=Dataset.TDBRAIN,
        modalities=("rtms",),
        db=TDBRAIN_DB,
        model=MODELS_DIR / "tdbrain_rtms_v1.pt",
        caption="Cohorte réelle TDBRAIN. " + _CLINICAL_CAPTION,
    ),
    Variant.SIM_MULTI: VariantConfig(
        key=Variant.SIM_MULTI,
        label="Données simulées — rTMS, EEG, ECG",
        dataset=Dataset.SIMULE,
        modalities=("rtms", "eeg", "ecg"),
        db=SIM_DB,
        model=MODELS_DIR / "sim_multi_v1.pt",
        caption="Cohorte simulée, calibrée sur TDBRAIN. " + _MULTI_CAPTION,
    ),
    Variant.TDBRAIN_MULTI: VariantConfig(
        key=Variant.TDBRAIN_MULTI,
        label="TDBRAIN — rTMS, EEG, ECG",
        dataset=Dataset.TDBRAIN,
        modalities=("rtms", "eeg", "ecg"),
        db=TDBRAIN_DB,
        model=MODELS_DIR / "tdbrain_multi_v1.pt",
        caption="Cohorte réelle TDBRAIN. " + _MULTI_CAPTION,
    ),
}

_REG_CAPTION_EEG = (
    "**Le modèle aligné sur l'article.** 130 puissances de bande (26 canaux × 5), "
    "et *aucune* variable clinique — l'étude de référence n'en donne aucune à son "
    "modèle. C'est la seule variante dont le r se compare à son r = 0.401."
)
_REG_CAPTION_MULTI = (
    "Cliniques + EEG + HRV. À lire avec précaution : le BDI-II de référence est "
    "ici une **entrée** du modèle, donc un bon r peut ne refléter que la sévérité "
    "initiale. Comparez « EEG seul » à « Clinique seul » pour trancher."
)
_REG_CAPTION_CLIN = (
    "**Référence à battre.** Mêmes cible et protocole, mais uniquement les "
    "variables cliniques — dont le BDI-II de référence, qui corrèle à lui seul "
    "r = 0.500 avec `delta_bdi` sur le protocole 1. Un modèle EEG qui ne "
    "dépasse pas ce chiffre n'a rien apporté."
)


def _regression_variants() -> dict[Variant, VariantConfig]:
    """The ten article-aligned entries, built from one spec table.

    Written as a loop rather than ten literals because every field except the
    cohort, protocol and feature set is identical across them — and a typo in
    one hand-written `db=` would train a checkpoint on the wrong cohort.
    """
    spec = (
        # (variant, protocol, modalities)
        (Variant.TDBRAIN_P1_EEG_REG, 1, ("eeg",)),
        (Variant.TDBRAIN_P2_EEG_REG, 2, ("eeg",)),
        (Variant.TDBRAIN_P1_CLIN_REG, 1, ("rtms",)),
        (Variant.TDBRAIN_P2_CLIN_REG, 2, ("rtms",)),
        (Variant.TDBRAIN_P1_MULTI_REG, 1, ("rtms", "eeg", "ecg")),
        (Variant.TDBRAIN_P2_MULTI_REG, 2, ("rtms", "eeg", "ecg")),
    )
    libelle = {
        ("eeg",): "EEG seul (130)",
        ("rtms",): "Clinique seul (4)",
        ("rtms", "eeg", "ecg"): "Multimodal (139)",
    }
    caption = {
        ("eeg",): _REG_CAPTION_EEG,
        ("rtms",): _REG_CAPTION_CLIN,
        ("rtms", "eeg", "ecg"): _REG_CAPTION_MULTI,
    }
    out: dict[Variant, VariantConfig] = {}
    for variant, protocol, modalities in spec:
        out[variant] = VariantConfig(
            key=variant,
            label=libelle[modalities],
            dataset=Dataset.TDBRAIN,
            modalities=modalities,
            db=TDBRAIN_DB,
            model=MODELS_DIR / f"{variant.value}_v1.pt",
            caption=caption[modalities],
            task=Task.REGRESSION,
            target="delta_bdi",
            protocol=protocol,
        )
    return out


VARIANTS.update(_regression_variants())

# The order the app presents. Clinical models first, then multimodal, alternating
# cohorts so the simulated/real pair sits side by side within each feature set.
ORDERED: tuple[Variant, ...] = (
    Variant.SIM_RTMS,
    Variant.TDBRAIN_RTMS,
    Variant.SIM_MULTI,
    Variant.TDBRAIN_MULTI,
)


def variant_config(variant: Variant) -> VariantConfig:
    return VARIANTS[variant]


# The article-aligned arm, in reporting order: real cohort first, and within each
# cohort the multimodal model immediately above the clinical baseline it must beat.
# EEG-only first: it is the article's own model, and the comparison that matters
# is the row immediately below it (clinical alone).
ARTICLE_ORDERED: tuple[Variant, ...] = (
    Variant.TDBRAIN_P1_EEG_REG,
    Variant.TDBRAIN_P1_CLIN_REG,
    Variant.TDBRAIN_P1_MULTI_REG,
    Variant.TDBRAIN_P2_EEG_REG,
    Variant.TDBRAIN_P2_CLIN_REG,
    Variant.TDBRAIN_P2_MULTI_REG,
)

ALL_ORDERED: tuple[Variant, ...] = ORDERED + ARTICLE_ORDERED


def variants_for(
    dataset: Dataset,
    task: Task | None = Task.CLASSIFICATION,
    protocol: int | None = None,
) -> tuple[VariantConfig, ...]:
    """Variants sharing a cohort — they can be trained from one data load.

    ``task`` defaults to classification so existing callers keep seeing exactly
    the original 2x2; pass ``None`` for every task. ``protocol`` filters to one
    arm — note that ``None`` here means *no filter*, while a variant's own
    ``protocol=None`` means *pooled*, so the pooled entries are only returned
    when no filter is applied.
    """
    return tuple(
        VARIANTS[v] for v in ALL_ORDERED
        if VARIANTS[v].dataset is dataset
        and (task is None or VARIANTS[v].task is task)
        and (protocol is None or VARIANTS[v].protocol == protocol)
    )


def article_variants(dataset: Dataset | None = None) -> tuple[VariantConfig, ...]:
    """The ten regression entries mirroring the reference study."""
    return tuple(
        VARIANTS[v] for v in ARTICLE_ORDERED
        if dataset is None or VARIANTS[v].dataset is dataset
    )
