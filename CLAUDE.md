# CLAUDE.md — guidance for future sessions

**The project is called NeuroRéponse** (repo `NeuroReponse`, ASCII because GitHub
mangles accents). It was previously "Recherche-App", which is still the name of
the working directory on disk — do not rename the folder to match, and do not
"fix" paths that legitimately contain it.

## Project state

**Phases 1–6 complete.** This is a finished PFE prototype: tests pass
(`pytest tests/ -q`), `ruff check src/` is clean, three executable notebooks,
working Streamlit app with PDF export.
Source of truth for the design remains `Description et Conception (Version0) (3).docx`.

**Real-data track — the app now runs on real EEG + ECG.** `src/data/tdbrain.py` loads
the public TDBRAIN rTMS-in-MDD cohort into a `LoadedDataset`; `src/data/tdbrain_seeder.py`
seeds it into SQLite, and the Streamlit app has a **data-source selector** (sidebar)
switching between the simulated cohort and TDBRAIN. See `docs/tdbrain.md`.
The track is **EEG + ECG** and **baseline-only**: no ERP, and no rTMS-session
trajectory, so the LSTM's sequence axis is filled by *epochs* of the resting recording.
Label = binary responder (≥50% BDI-II reduction). The data is gated + git-ignored;
the loader/seeder are validated against a synthetic fixture (`make_synthetic_tdbrain`,
which can emit a synthetic `Erbs` lead via `with_ecg=True`).

Key facts to not re-derive:

- **There are three cohorts, in three separate SQLite files**, and they must never
  share a table. See `SOURCES` in `src/app/utils.py`:
  - `recherche.sqlite3` — the **legacy sequential** simulator (100 patients × 10
    *treatment sessions* × 1 channel), model `lstm_v1.pt`. The only cohort where
    the LSTM accumulates evidence across sessions; page 6 depends on it.
  - `recherche_sim_matched.sqlite3` — the **matched** simulator
    (`simulate_matched()`, 132 × 8 epochs × 26 ch + ECG), models `sim_rtms_v1` /
    `sim_multi_v1`. Seed with `python -m src.data.tdbrain_seeder --matched`.
  - `recherche_tdbrain.sqlite3` — the real cohort, `tdbrain_rtms_v1` /
    `tdbrain_multi_v1`.
  The first two are both "simulated" but are **different cohorts of different
  shape**; `variants.SIM_DB` points at the *matched* one, because that is what the
  2×2's simulated checkpoints were fit on.
- **The app serves the whole 2×2.** The sidebar picks cohort × feature set
  (`source_selector`), `variants.py` is the registry both training and the app
  read, and `7_Comparaison.py` renders `comparison.json`. Nothing in `src/app/`
  hard-codes a checkpoint path.
- **`Patient.age` is a float and `Patient.sexe` exists** (0/1, nullable). Both are
  load-bearing, not cosmetic: the clinical block is
  `(rtms_protocol, age, gender, bdi_pre)`, TDBRAIN publishes decimal ages (49.66),
  and age is the strongest predictor in the project. Rounding it on the way into
  SQLite made the app predict on a value the model never saw. The legacy
  sequential cohort has no gender, so its patients carry `sexe=None` — which
  `clinical_block` **refuses** rather than imputing (training imputes with the
  *cohort* median, which one patient cannot compute).
- **`dataset_from_repository`** (`src/data/tdbrain_seeder.py`) rebuilds a full
  `LoadedDataset` — montage, ECG *and* the clinical metadata — from the database,
  so the Training page can call the same `build_features` training calls.
  Verified exact: DB round-trip reproduces the 139-column training tensor with
  max abs diff 0.0. `montage_from_repository` is now a thin wrapper on it.
- **Two incompatible feature contracts.** Simulated = 8 features from *one* channel
  (`preprocessing.pipeline.preprocess`). TDBRAIN = **130** features (26 channels × 5
  bands, `tdbrain.montage_band_powers`). Every TDBRAIN checkpoint therefore ships a
  **JSON sidecar** (`data/models/*.json`) recording fs/channels/n_epochs/z-scoring;
  `4_Predictions.py` rebuilds inputs from it and refuses to predict on a mismatch.
  Selecting channels **by name** in the sidecar's order is what keeps the DB's row
  order from silently permuting the input vector.
- **Result (measured, 132 patients, 5-fold patient-wise, 4 variants):** rTMS response
  is **at chance** whatever you feed it — AUC 0.465 (EEG raw), 0.467 (EEG z-scored),
  0.516 (EEG+ECG raw), 0.500 (EEG+ECG z-scored). The +0.050 from HRV is smaller than
  the fold std (±0.078), and **accuracy 0.628 / F1 0.768 are identical across all
  four** — that is exactly the all-positive predictor (base rate 83/132). The model
  predicts "responder" for everyone; the AUC wobble is noise. Normalisation is *not*
  the bottleneck for response (it was for the MDD-vs-control diagnosis task, where
  raw band power + logistic regression reaches 0.86). Treat single-baseline response
  prediction as a known-hard negative result, not a bug to fix — and do not present
  the ECG delta as a gain.
- `RTMSParameters` for TDBRAIN: protocol 1 = 10 Hz L-DLPFC, protocol 2 = 1 Hz R-DLPFC
  (real). Stimulation intensity/train counts are **not published** and are stored as
  `0.0` with a `protocole` string saying so — do not backfill them with plausible
  numbers. Consequence: `total_pulses()` is 0 for every TDBRAIN patient.
- **Modality inventory (measured, don't re-derive).** TDBRAIN has EEG, ECG and ERP,
  but **rTMS outcome and ERP never co-occur**: all 190 subjects with an rTMS protocol
  have `restEO`/`restEC` only, and all 129 subjects with the `oddball` ERP task are
  healthy controls. The joint maximum is **rTMS + EEG + ECG = 190** (163 MDD).
  A four-modality model on TDBRAIN is impossible, not merely unimplemented.
- **rTMS parameters carry one bit and it is null.** `frequence_hz`, `localisation`
  and `protocole` are all deterministic functions of the protocol integer.
  Responder rate is 61.4% (proto 1) vs 64.4% (proto 2): χ² p=0.885, AUC 0.514,
  MI 0.0004 nats — and protocol is confounded with age (p=0.013). Use it to
  **stratify**, never as a feature.
- **Clinical loop (`6_Boucle_clinique.py`)** — record a session → predict → adjust
  the stimulator externally → repeat. Logic in `src/reporting/boucle.py` +
  `src/reporting/enregistrement.py`; tests in `tests/test_boucle.py` and
  `tests/test_boucle_tdbrain.py`. **Both sources work, by different mechanisms:**
  - *Simulated* = **sequential**. The model eats a sequence of treatment sessions,
    so each new one lengthens its input; `predict_tri()[k]` *is* the estimate after
    session k+1. Accumulation happens inside the LSTM.
  - *TDBRAIN* = **snapshot**. The model is baseline-only (n_epochs windows of ONE
    resting recording), so each session re-records and predicts on that recording
    alone → one independent probability per session; the trend is clinical, not
    recurrent. Do **not** "fix" this by relaxing `FeatureContract.matches()`'s
    n_epochs check to feed sessions as timesteps — that is a distribution the
    model never saw, and the check is the thing preventing it.
  - Uploaded recordings go through `lire_enregistrement`, which delegates to the
    *training* loader (same notch/band-pass/resample/R-peak detection) and applies
    `_tachogram(rr, contract.n_rr)`. That truncation matters: HRV over 140 beats
    ≠ HRV over the first 64, and skipping it silently trains on one autonomic
    feature and predicts on another. `test_snapshot_matches_what_the_training_loader_would_produce`
    pins the two paths together — it caught exactly this bug.
  - `recommandation()` reports *direction*, never a suggested intensity: no data
    in this project links dose to response.
  - The TDBRAIN arm carries an on-screen validity warning. The model is at chance,
    so the per-session curve is flat (~0.66 regardless of input). The loop
    demonstrates the **workflow**, not a usable clinical signal.
- **Follow-up page (`5_Suivi.py`) reads *all* sessions.** Logic lives in
  `src/reporting/suivi.py` (`analyser_suivi`), not the page, so it is testable
  (`tests/test_suivi.py`). It detects whether `score_post` actually varies:
  the simulated cohort has a real 10-session trajectory, TDBRAIN's epochs all
  carry the same scores. When it doesn't vary the synthesis **refuses to report a
  trend** and reframes the TRI spread as epoch-to-epoch *coherence*, not progress.
  Don't "fix" that by fitting a slope over TDBRAIN epochs — it would be fiction.
- **Model inputs are rebuilt in one place**, `src/app/inference.py`
  (`build_model_input`). Predictions and Suivi both call it; duplicating the
  reconstruction is how the two pages start disagreeing about the same patient.
- **Unpublished rTMS params render as "— (non publié)"** via
  `format_rtms_parameters` in `src/app/utils.py`. A stored `0.0` means *not
  reported*, never *measured zero* — displaying the raw 0 (and `total_pulses` = 0)
  read as a real measurement.
- **ECG is real here.** Every recording carries an `Erbs` lead; `detect_rr_intervals`
  (Pan–Tompkins + physiological gate + Malik filter) turns it into an RR tachogram.
  HRV is **patient-level** — measured over the full ~120 s and repeated across
  epochs, because an 8 s epoch holds only ~9 beats. Therefore the HRV block is
  **never per-patient z-scored**: it has zero within-patient variance and would
  collapse to constant 0. `tdbrain_features(modalities=("eeg","ecg"))` z-scores the
  EEG block only. Guard: `test_zscoring_does_not_blank_the_hrv_block`.

Real EEG is **BioSemi BDF** (`*_eeg.bdf`), read via `mne` (now a dependency); the loader
applies a configurable power-line notch + band-pass (`TDBRAINConfig.notch_hz`/`bandpass_hz`)
because raw BDF is unfiltered. `make_synthetic_tdbrain(..., fmt="bdf")` writes a BDF tree
for tests (needs `mne`+`edfio`; those tests `importorskip`). **Caveat:** the public
"Treatment/Diagnostic Prediction" packages are *blinded* (labels = `REPLICATION`); real
responder labels live only in the full `TDBRAIN Dataset V3.1` download. `download_tdbrain.py`
is a resume-capable fetcher for that ~14.5 GB encrypted zip.

## The four-model comparison (2×2)

`src/models/variants.py` defines the four models once — training and the app both
import it. Two cohorts × two feature sets:

| variant | cohort | features | AUC (5-fold, patient-wise) |
|---|---|---|---|
| `sim_rtms` | simulated | 4 (clinical) | **0.629** ± 0.115 |
| `tdbrain_rtms` | real | 4 (clinical) | **0.574** ± 0.111 |
| `sim_multi` | simulated | 139 | 0.582 ± 0.173 |
| `tdbrain_multi` | real | 139 | 0.488 ± 0.087 |

Facts worth not re-deriving:

- **The clinical model beats the multimodal one on both cohorts.** 4 features beat
  139. Adding 135 uninformative columns dilutes the 4 informative ones on 132
  patients.
- **Age is the only real predictor found anywhere in this project** — AUC 0.610
  alone on the real cohort, and it dominates permutation importance (+0.128).
  `rtms_protocol` scores *negative* (−0.020). Baseline BDI-II contributes almost
  nothing (30.8 vs 32.4 across classes).
- **No variant beats its base rate on accuracy.** All sit at ~0.63, the
  majority-class rate; the confusion matrices show an empty non-responder column.
- **`simulator_matched` is a positive control, not decoration.** Do not delete the
  effect knob. **But the curve only appears without per-patient z-scoring**, and
  the previously recorded figures (0.51 → 0.60/0.80/0.97) do **not** reproduce
  through `build_features`' default path. Measured by
  `python -m src.reporting.effect_sweep` (`docs/strategy/effect_sweep.csv`):
  - *EEG **raw*** — 0.582 / 0.594 / 0.634 / **0.730** at effect 0 / 0.15 / 0.30 /
    0.50. Climbs. This is the real positive control.
  - *EEG **z-scored*** — 0.468 / 0.467 / 0.467 / **0.463**. Dead flat, including
    at effect 0.50.
  - *Multimodal z-scored* — 0.582 / 0.624 / 0.661 / 0.695.
  - The simulator raises responders' alpha by a **per-patient constant**, applied
    identically to every epoch. `zscore_epochs` centres each patient on their own
    epochs, so it subtracts precisely the quantity carrying the label. Same
    mechanism as the HRV block — a between-patient effect cannot survive
    within-patient normalisation.
  - Consequence: the multimodal arm's climb comes from the **HRV** block (which is
    never z-scored), not from EEG.
  - `tests/test_effect_sweep.py` pins the mechanism directly on the features, so
    it stays true without retraining.
  - This does **not** overturn the negative result: on *real* data both raw and
    z-scored EEG are at chance (0.465 / 0.467). It means the z-scored simulated
    arm was never able to demonstrate detection either way.
- Two different AUCs appear in the reports: **mean of per-fold AUCs** (verdict
  panel) and **pooled out-of-fold AUC** (ROC curve). Both are legitimate and they
  do not match; the labels say which is which.

## Conventions

- **Language**: French domain terms (`ajouter_dossier`, `frequence_hz`,
  `localisation`) because the UML diagram uses French. The jury looks for these.
  Python identifiers use snake_case (no accents).
- **ML backend**: **PyTorch**, not TensorFlow — TF doesn't ship wheels for
  Python 3.14. Architecture and metrics are identical to what the doc describes.
- **Persistence**: SQLAlchemy 2.0 ORM in `src/db/schema.py`; all DB access goes
  through `Repository` (= the `BaseDeDonnées` UML class).
- **Cross-validation**: always patient-wise (`GroupKFold` with patient id as
  group). The test `test_groupkfold_keeps_patients_separated` enforces this.
- **Normalization**: per-patient, never across patients. The test
  `test_pipeline_no_cross_patient_leakage` enforces this.

## Testing

- `pytest tests/ -q` should always pass (201 tests).
- `tests/test_app_pages.py` runs every Streamlit page through `AppTest`. Pages are
  scripts nothing imports, so a rename in `utils` is otherwise invisible until
  someone clicks the page. It **skips** when a cohort's database is absent, so it
  stays green on a machine that has never run the seeders.
- `python -m ruff check src/ --select=E,F,I,UP --ignore=E501` must be clean.
- `python -m jupyter nbconvert --to notebook --execute notebooks/0X_*.ipynb`
  must execute **all** notebooks end-to-end without error. Notebooks 05–08 are
  **generated** by `python -m src.reporting.build_notebooks` — edit the builder,
  never the `.ipynb`, or the next regeneration silently discards the change.
  06/08 fall back to the synthetic fixture when the gated TDBRAIN data is absent,
  so they stay executable on any machine.

## Things not to do

- Do **not** introduce a web framework, microservices, or auth — out of scope.
- Do **not** drop the French method/attribute names to "tidy them up".
- Do **not** add real patient data to the repo. `data/simulated/` is for
  generator output only and is gitignored.
- Do **not** mock the DB in tests; SQLite is already free and exact.
- Do **not** re-introduce TensorFlow without checking whether it now ships
  Python 3.14 wheels.

## How to extend

- **Add a new feature** (e.g. fNIRS HbO/HbR signals): add a new `SignalType`
  enum value, extend the simulator with a second channel type, expand
  `basic_features` to compute relevant fNIRS features. The pipeline contract
  `(n_patients, n_sessions, n_features)` stays unchanged.
- **Add an attention layer**: see `LSTMConfig.bidirectional` — add a similar
  `use_attention` flag, wire it in `ResponseLSTM.__init__`. The doc explicitly
  calls this out as the next iteration.
- **Swap the simulated dataset for a real one**: implement a new loader that
  returns a `LoadedDataset` from `src/data/loader.py`. Everything downstream
  (preprocessing, training, app) will work unchanged. `src/data/tdbrain.py` is the
  worked example (real TDBRAIN EEG → `LoadedDataset` + `tdbrain_features` →
  `cross_validate`); `docs/tdbrain.md` documents the format and caveats. Note the
  new optional `LoadedDataset` fields `channels` / `signals_mc` carry a full EEG
  montage when a real multi-channel source provides one.
- **Add EMG/ERP**: same as fNIRS above. Note TDBRAIN also ships 4 EOG channels and
  an EMG lead (`Mass`) in the same BDF; `_read_recording` is where you would pick
  them up. ERP is *not* available for treated patients (see the modality inventory).
- **Add a modality to the TDBRAIN track**: `tdbrain_features(modalities=...)` is the
  seam. Add the block there, extend `FeatureContract.modalities`, persist it in
  `tdbrain_seeder` as its own `SignalType`, and rebuild it identically in
  `4_Predictions.py`. Ask whether the new block varies across epochs — if it does
  not, keep it out of `zscore_epochs` the way HRV is.

## Useful commands

```powershell
python -m pytest -q                          # tests
python -m ruff check src/                    # lint
python -m src.data.simulator                 # regenerate simulated data
python -m src.data.seeder                    # seed SQLite from simulated data

# Real TDBRAIN track (reads BDF — minutes, not seconds)
python -m src.data.tdbrain_seeder --root "data/tdbrain/TDBRAIN_Dataset_V3_1_Encr/TDBRAIN_Dataset_V3_1" `
    --db recherche_tdbrain.sqlite3           # seed the real cohort (132 patients)
python -m src.data.tdbrain_seeder --matched  # seed the matched simulated cohort

python -m src.models.train_tdbrain --root "data/tdbrain/TDBRAIN_Dataset_V3_1_Encr/TDBRAIN_Dataset_V3_1" `
    --seed-db recherche_tdbrain.sqlite3      # load once: seed + CV all variants + save model
# --modalities eeg | eeg+ecg | auto (default: evaluate both, keep the better)
# --no-ecg   skip the autonomic lead entirely (faster load)

python -m src.reporting.figures              # regenerate README figures
streamlit run src/app/main.py                # launch app
jupyter nbconvert --to notebook --execute notebooks/03_results_analysis.ipynb \
    --output 03_results_analysis.ipynb       # re-run results notebook
```
