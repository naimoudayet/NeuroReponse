"""Train the response LSTM on the **real** TDBRAIN cohort and persist it for the app.

Two things distinguish this from the simulated training path:

1. **Feature contract.** The simulated pipeline feeds 8 features from a single
   channel (:func:`src.preprocessing.pipeline.preprocess`). TDBRAIN gives a
   26-channel montage, so the model consumes 26 x 5 = 130 relative band powers
   (:func:`src.data.tdbrain.montage_band_powers`). A model trained on one
   contract cannot read the other, so every saved model gets a **JSON sidecar**
   recording exactly how its inputs were built. The app refuses to predict when
   the sidecar does not match the data it holds.

2. **Normalisation is chosen, not assumed.** Per-patient z-scoring was inherited
   from the simulated multi-session design, where it removes a per-patient scale
   nuisance across *treatment sessions*. TDBRAIN has one recording, so the
   "sessions" are epochs of it, and z-scoring across them deletes the patient's
   absolute spectral profile instead. Which is better is an empirical question,
   so both are cross-validated here and the winner is persisted.

Note on the metric: :func:`cross_validate` early-stops on the same fold it
scores, so the reported AUC is optimistic in absolute terms. It is used here
only to *rank* two feature variants under an identical protocol, which is a fair
comparison.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data.loader import LoadedDataset
from ..data.tdbrain import BANDS, tdbrain_features
from ..models.lstm import LSTMConfig
from ..models.train import TrainConfig, cross_validate, fit_final_model, save_model

DEFAULT_MODEL_PATH = Path("data/models/tdbrain_response_v1.pt")


def sidecar_path(model_path: Path) -> Path:
    """Companion JSON describing how a checkpoint's inputs were built."""
    return model_path.with_suffix(".json")


@dataclass
class FeatureContract:
    """Everything needed to rebuild this model's inputs at inference time."""

    source: str
    task: str
    features: str
    fs: float
    channels: list[str]
    n_bands: int
    per_patient_zscore: bool
    input_size: int
    window: int
    n_epochs: int
    # Multimodal fields. Defaulted so that sidecars written before the ECG track
    # existed still load as EEG-only contracts instead of raising.
    modalities: list[str] = field(default_factory=lambda: ["eeg"])
    ecg_channel: str | None = None
    n_rr: int = 0
    # Article-aligned axes (Arteaga et al., PMC12981298). Defaulted so every
    # sidecar written before the regression arm existed still loads as the
    # pooled binary-responder contract it was.
    #
    # `target` is the single source of truth for which head the checkpoint has:
    # storing a separate "regression" flag alongside it would let the two
    # disagree, and a contract that contradicts itself is worse than one field.
    target: str = "responder"
    protocol: int | None = None

    @property
    def is_regression(self) -> bool:
        return self.target != "responder"

    @property
    def target_label(self) -> str:
        return {
            "responder": "réponse (binaire, ≥50 % de réduction)",
            "delta_bdi": "réduction BDI-II (points)",
            "pct_reduction": "réduction BDI-II (%)",
        }.get(self.target, self.target)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def matches(
        self,
        available_channels,
        fs: float,
        n_epochs: int,
        has_ecg: bool = False,
    ) -> tuple[bool, str]:
        """Check stored data against this contract; returns ``(ok, reason)``.

        ``available_channels`` is whatever the data offers, in any order — the
        contract only requires that every channel it needs is present, since the
        caller selects them by name into the contract's order. ``has_ecg`` reports
        whether the stored patient carries an autonomic tachogram; a model trained
        with the ECG block cannot run without it (the feature vector would be 5
        columns short), so that mismatch is refused rather than zero-filled.
        """
        # A clinical-only model reads nothing from the recordings, so montage and
        # sampling-rate checks do not apply to it — enforcing them would reject a
        # patient whose EEG is irrelevant to the prediction.
        if "eeg" in self.modalities:
            have = set(available_channels)
            missing = [c for c in self.channels if c not in have]
            if missing:
                return False, (
                    f"canaux absents en base : {missing[:5]}"
                    f"{'…' if len(missing) > 5 else ''} "
                    f"(le modèle en attend {len(self.channels)})"
                )
            if abs(float(fs) - float(self.fs)) > 1e-6:
                return False, (
                    f"fréquence d'échantillonnage : modèle {self.fs} Hz, base {fs} Hz"
                )
        if self.uses_signals and int(n_epochs) != int(self.n_epochs):
            return False, f"nombre d'époques : modèle {self.n_epochs}, base {n_epochs}"
        if "ecg" in self.modalities and not has_ecg:
            return False, (
                "le modèle attend la modalité ECG (HRV) mais aucun signal ECG "
                "n'est enregistré pour ce patient"
            )
        return True, "ok"

    @property
    def uses_signals(self) -> bool:
        """Whether this contract consumes recordings rather than metadata alone."""
        return bool({"eeg", "ecg"} & set(self.modalities))


def load_contract(model_path: Path) -> FeatureContract | None:
    path = sidecar_path(model_path)
    if not path.exists():
        return None
    return FeatureContract(**json.loads(path.read_text(encoding="utf-8")))


def available_modality_sets(dataset: LoadedDataset) -> list[tuple[str, ...]]:
    """Modality combinations this dataset can actually support, cheapest first."""
    sets: list[tuple[str, ...]] = [("eeg",)]
    if dataset.ecg is not None:
        sets.append(("eeg", "ecg"))
    return sets


def evaluate_variants(
    dataset: LoadedDataset,
    n_splits: int = 5,
    train_cfg: TrainConfig | None = None,
    modality_sets: list[tuple[str, ...]] | None = None,
) -> dict[tuple[tuple[str, ...], bool], dict]:
    """Cross-validate every (modality set x normalisation) combination.

    Returns ``{(modalities, per_patient_zscore): summary}``. Running EEG alone
    alongside EEG+ECG is the point: it turns "we added the autonomic channel" into
    a measured delta rather than an assertion, which is the ablation the NPDT
    design asks for.
    """
    if modality_sets is None:
        modality_sets = available_modality_sets(dataset)
    out: dict[tuple[tuple[str, ...], bool], dict] = {}
    for mods in modality_sets:
        for zscore in (False, True):
            x, y, groups, _ = tdbrain_features(
                dataset, per_patient_zscore=zscore, modalities=mods
            )
            cv = cross_validate(
                x, y.astype(np.float32), groups,
                lstm_cfg=LSTMConfig(input_size=x.shape[-1]),
                train_cfg=train_cfg or TrainConfig(),
                n_splits=n_splits,
            )
            out[(mods, zscore)] = cv.summary()
    return out


def train_and_save(
    dataset: LoadedDataset,
    model_path: Path = DEFAULT_MODEL_PATH,
    per_patient_zscore: bool | None = None,
    n_splits: int = 5,
    train_cfg: TrainConfig | None = None,
    modalities: tuple[str, ...] | None = None,
    ecg_channel: str | None = None,
) -> tuple[FeatureContract, dict[tuple[tuple[str, ...], bool], dict]]:
    """Pick the best (modalities, normalisation) unless forced, fit, save.

    Writes ``model_path`` plus its JSON sidecar. Returns the contract and the
    cross-validation summaries for every evaluated variant.
    """
    if dataset.signals_mc is None or not dataset.channels:
        raise ValueError("need a full montage (signals_mc + channels) from load_tdbrain()")
    if modalities is not None and "ecg" in modalities and dataset.ecg is None:
        raise ValueError("modalities include 'ecg' but the dataset carries no ECG tachogram")

    summaries = evaluate_variants(
        dataset, n_splits=n_splits, train_cfg=train_cfg,
        modality_sets=[tuple(modalities)] if modalities else None,
    )
    best = max(summaries.items(), key=lambda kv: kv[1]["auc_mean"])[0]
    chosen_mods = tuple(modalities) if modalities is not None else best[0]
    if per_patient_zscore is None:
        # Compare normalisations *within* the chosen modality set, so the choice is
        # not confounded by a modality difference.
        same = {k: v for k, v in summaries.items() if k[0] == chosen_mods}
        per_patient_zscore = max(same.items(), key=lambda kv: kv[1]["auc_mean"])[0][1]

    x, y, _, names = tdbrain_features(
        dataset, per_patient_zscore=per_patient_zscore, modalities=chosen_mods
    )
    lstm_cfg = LSTMConfig(input_size=x.shape[-1])
    model, _, _ = fit_final_model(
        x, y.astype(np.float32), lstm_cfg=lstm_cfg, train_cfg=train_cfg or TrainConfig()
    )
    save_model(model, model_path)

    contract = FeatureContract(
        source="tdbrain",
        task="response",
        features="montage_band_powers+hrv" if "ecg" in chosen_mods else "montage_band_powers",
        fs=float(dataset.fs),
        channels=list(dataset.channels),
        n_bands=len(BANDS),
        per_patient_zscore=bool(per_patient_zscore),
        input_size=int(x.shape[-1]),
        window=int(dataset.window),
        n_epochs=int(x.shape[1]),
        modalities=list(chosen_mods),
        ecg_channel=ecg_channel if "ecg" in chosen_mods else None,
        n_rr=int(dataset.ecg.shape[-1]) if "ecg" in chosen_mods and dataset.ecg is not None else 0,
    )
    sidecar_path(model_path).write_text(
        json.dumps(contract.to_dict(), indent=2), encoding="utf-8"
    )
    return contract, summaries


def _main() -> None:
    import argparse
    import warnings

    from ..data.tdbrain import TDBRAINConfig, load_tdbrain

    ap = argparse.ArgumentParser(
        description="Train + persist the response LSTM on real TDBRAIN EEG."
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, default=None)
    ap.add_argument("--col-id", default="TDBRAIN_ID")
    ap.add_argument("--col-protocol", default="rTMS PROTOCOL")
    ap.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--zscore", choices=("auto", "on", "off"), default="auto",
                    help="per-patient z-scoring; 'auto' picks the better by CV AUC")
    ap.add_argument("--modalities", default="auto",
                    help="'auto' (evaluate eeg and eeg+ecg, keep the better), "
                         "'eeg', or 'eeg+ecg'")
    ap.add_argument("--no-ecg", action="store_true",
                    help="skip reading the autonomic lead entirely (faster load)")
    ap.add_argument("--seed-db", type=Path, default=None,
                    help="also seed this SQLite file from the same load (avoids re-reading BDF)")
    args = ap.parse_args()

    cfg = TDBRAINConfig(root=args.root, metadata_path=args.metadata,
                        col_id=args.col_id, col_protocol=args.col_protocol,
                        ecg_channel=None if args.no_ecg else "Erbs")
    print(f"loading TDBRAIN from {args.root} …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = load_tdbrain(cfg)
    print(f"  {ds.signals_mc.shape[0]} patients · montage {ds.signals_mc.shape} · fs {ds.fs} Hz")
    print(f"  responders {int(ds.labels.sum())}/{len(ds.labels)}")
    print(f"  ECG tachogram: {'absent' if ds.ecg is None else ds.ecg.shape}")

    if args.seed_db is not None:
        from ..data.tdbrain_seeder import seed_tdbrain
        from ..db import Repository

        repo = Repository(db_url=f"sqlite:///{args.seed_db}")
        n = seed_tdbrain(repo, ds, progress=lambda d, t: print(f"  seeding {d}/{t}", end="\r"))
        print(f"\n  seeded {n} patients into {args.seed_db}")

    forced = {"auto": None, "on": True, "off": False}[args.zscore]
    forced_mods = None if args.modalities == "auto" else tuple(args.modalities.split("+"))
    contract, summaries = train_and_save(
        ds, model_path=args.out, per_patient_zscore=forced, n_splits=args.n_splits,
        modalities=forced_mods, ecg_channel=cfg.ecg_channel,
    )

    print("\n===== cross-validation (patient-wise GroupKFold) =====")
    for (mods, z), s in sorted(summaries.items(), key=lambda kv: (len(kv[0][0]), kv[0][1])):
        tag = f"{'+'.join(mods):<8} {'z-scored' if z else 'raw     '}"
        print(f"  {tag}  AUC {s['auc_mean']:.3f} +/- {s['auc_std']:.3f}"
              f"   acc {s['accuracy_mean']:.3f}   F1 {s['f1_mean']:.3f}")

    # The headline number for the multimodal claim: does the autonomic channel add
    # anything over EEG alone, holding normalisation fixed?
    z = contract.per_patient_zscore
    if (("eeg",), z) in summaries and (("eeg", "ecg"), z) in summaries:
        delta = summaries[(("eeg", "ecg"), z)]["auc_mean"] - summaries[(("eeg",), z)]["auc_mean"]
        print(f"\n  ECG contribution (same normalisation): dAUC = {delta:+.3f}")

    print(f"\nchosen : modalities={'+'.join(contract.modalities)} "
          f"per_patient_zscore={contract.per_patient_zscore} "
          f"input_size={contract.input_size}")
    print(f"saved  : {args.out}")
    print(f"sidecar: {sidecar_path(args.out)}")


if __name__ == "__main__":
    _main()
