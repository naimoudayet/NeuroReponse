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


class Variant(str, Enum):
    SIM_RTMS = "sim_rtms"
    TDBRAIN_RTMS = "tdbrain_rtms"
    SIM_MULTI = "sim_multi"
    TDBRAIN_MULTI = "tdbrain_multi"


@dataclass(frozen=True)
class VariantConfig:
    key: Variant
    label: str                       # shown in the app's source selector
    dataset: Dataset
    modalities: tuple[str, ...]
    db: Path
    model: Path
    caption: str

    @property
    def sidecar(self) -> Path:
        return self.model.with_suffix(".json")

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


def variants_for(dataset: Dataset) -> tuple[VariantConfig, ...]:
    """All variants sharing a cohort — they can be trained from one data load."""
    return tuple(VARIANTS[v] for v in ORDERED if VARIANTS[v].dataset is dataset)
