"""Tests for the four-model registry and the training driver.

The registry is the single definition of what each model eats and where it lives.
If it drifts from what training actually writes, the app will load a checkpoint
and feed it the wrong feature vector — silently, because the shapes can still
line up. These tests pin the two together.
"""
from __future__ import annotations

import json

import pytest

from src.data.simulator_matched import MatchedSimConfig, simulate_matched
from src.models.train import TrainConfig
from src.models.train_all import comparison_table, train_variant
from src.models.train_tdbrain import load_contract
from src.models.variants import (
    ORDERED,
    VARIANTS,
    Dataset,
    Variant,
    variant_config,
    variants_for,
)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_there_are_exactly_four_variants():
    assert len(VARIANTS) == 4
    assert len(ORDERED) == 4
    assert set(ORDERED) == set(VARIANTS)


def test_presentation_order_is_the_one_the_app_must_use():
    assert ORDERED == (
        Variant.SIM_RTMS,
        Variant.TDBRAIN_RTMS,
        Variant.SIM_MULTI,
        Variant.TDBRAIN_MULTI,
    )
    labels = [VARIANTS[v].label for v in ORDERED]
    assert labels == [
        "Données simulées — rTMS",
        "TDBRAIN — rTMS",
        "Données simulées — rTMS, EEG, ECG",
        "TDBRAIN — rTMS, EEG, ECG",
    ]


def test_the_2x2_covers_both_cohorts_and_both_feature_sets():
    combos = {(c.dataset, c.modalities) for c in VARIANTS.values()}
    assert combos == {
        (Dataset.SIMULE, ("rtms",)),
        (Dataset.TDBRAIN, ("rtms",)),
        (Dataset.SIMULE, ("rtms", "eeg", "ecg")),
        (Dataset.TDBRAIN, ("rtms", "eeg", "ecg")),
    }


def test_each_variant_has_its_own_checkpoint():
    paths = [c.model for c in VARIANTS.values()]
    assert len(set(paths)) == 4
    assert all(p.suffix == ".pt" for p in paths)
    assert all(c.sidecar.suffix == ".json" for c in VARIANTS.values())


def test_the_two_cohorts_use_separate_databases():
    dbs = {c.dataset: c.db for c in VARIANTS.values()}
    assert dbs[Dataset.SIMULE] != dbs[Dataset.TDBRAIN]
    # ...but variants of the same cohort share one database.
    assert len({c.db for c in variants_for(Dataset.SIMULE)}) == 1
    assert len({c.db for c in variants_for(Dataset.TDBRAIN)}) == 1


def test_clinical_variants_need_no_signals():
    assert variant_config(Variant.SIM_RTMS).uses_signals is False
    assert variant_config(Variant.TDBRAIN_RTMS).uses_signals is False
    assert variant_config(Variant.SIM_MULTI).uses_signals is True
    assert variant_config(Variant.TDBRAIN_MULTI).uses_signals is True


def test_variants_for_returns_both_of_a_cohorts_models():
    sim = variants_for(Dataset.SIMULE)
    assert [c.key for c in sim] == [Variant.SIM_RTMS, Variant.SIM_MULTI]


# --------------------------------------------------------------------------- #
# Training writes what the registry promises
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tiny():
    return simulate_matched(
        MatchedSimConfig(n_patients=24, n_epochs=3, window=256, seed=9)
    )


@pytest.mark.parametrize("variant,expected_width", [
    (Variant.SIM_RTMS, 4),
    (Variant.SIM_MULTI, 4 + 130 + 5),
])
def test_training_writes_a_checkpoint_and_a_matching_contract(
    tmp_path, tiny, variant, expected_width
):
    cfg = variant_config(variant)
    # Redirect the artefacts into the test's own directory.
    from dataclasses import replace

    cfg = replace(cfg, model=tmp_path / cfg.model.name)

    result = train_variant(
        cfg, tiny, n_splits=2, train_cfg=TrainConfig(epochs=2, batch_size=8)
    )

    assert cfg.model.exists()
    assert cfg.sidecar.exists()
    assert result.n_features == expected_width
    assert result.n_patients == 24
    assert 0.0 <= result.auc_mean <= 1.0

    contract = load_contract(cfg.model)
    assert contract.input_size == expected_width
    assert contract.modalities == list(cfg.modalities)
    assert contract.source == cfg.dataset.value
    # A clinical-only contract must not claim EEG channels it never used.
    if "eeg" not in cfg.modalities:
        assert contract.channels == []
        assert contract.uses_signals is False


def test_clinical_contract_ignores_montage_and_epoch_checks(tmp_path, tiny):
    """A clinical model reads metadata only; rejecting a patient over channels or
    epoch count would be wrong."""
    from dataclasses import replace

    cfg = replace(variant_config(Variant.SIM_RTMS),
                  model=tmp_path / "clin.pt")
    train_variant(cfg, tiny, n_splits=2, train_cfg=TrainConfig(epochs=2, batch_size=8))
    contract = load_contract(cfg.model)

    ok, why = contract.matches(available_channels=set(), fs=999.0, n_epochs=99)
    assert ok, why


def test_multimodal_contract_still_enforces_its_requirements(tmp_path, tiny):
    from dataclasses import replace

    cfg = replace(variant_config(Variant.SIM_MULTI), model=tmp_path / "multi.pt")
    train_variant(cfg, tiny, n_splits=2, train_cfg=TrainConfig(epochs=2, batch_size=8))
    contract = load_contract(cfg.model)

    ok, why = contract.matches(available_channels=set(), fs=250.0, n_epochs=3)
    assert not ok and "canaux" in why

    ok, why = contract.matches(set(contract.channels), 250.0, 3, has_ecg=False)
    assert not ok and "ECG" in why

    ok, _ = contract.matches(set(contract.channels), 250.0, 3, has_ecg=True)
    assert ok


def test_comparison_table_renders_every_result(tmp_path, tiny):
    from dataclasses import replace

    results = [
        train_variant(
            replace(variant_config(v), model=tmp_path / f"{v.value}.pt"),
            tiny, n_splits=2, train_cfg=TrainConfig(epochs=1, batch_size=8),
        )
        for v in (Variant.SIM_RTMS, Variant.SIM_MULTI)
    ]
    table = comparison_table(results)
    assert "sim_rtms" in table and "sim_multi" in table
    assert "AUC" in table and "base" in table
    assert len(table.splitlines()) == 4          # header + rule + 2 rows


def test_comparison_results_serialise_to_json(tmp_path, tiny):
    from dataclasses import replace

    r = train_variant(
        replace(variant_config(Variant.SIM_RTMS), model=tmp_path / "a.pt"),
        tiny, n_splits=2, train_cfg=TrainConfig(epochs=1, batch_size=8),
    )
    payload = json.loads(json.dumps([r.to_dict()]))
    assert payload[0]["key"] == "sim_rtms"
    assert payload[0]["modalities"] == ["rtms"]
