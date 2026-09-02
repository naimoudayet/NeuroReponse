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
- **Session order is pinned in the schema, and that is load-bearing.**
  `PatientRow.sessions` carries `order_by="(SessionRow.date, SessionRow.id_session)"`
  (`src/db/schema.py`), and `lister_sessions_patient` repeats it. The LSTM's
  sequence axis **is** that list: `predict_tri()[k]` means "after session k+1",
  `etapes_boucle` reads the delta between consecutive entries, and
  `analyser_suivi` fits a slope over them. Without an ORDER BY, SQLite returns
  rowid (= insertion) order, which matched the seeders by luck and stopped
  matching the first time a missed visit was recorded late — reordering the
  treatment course with **no error anywhere**. Date first (clinical truth), id as
  tiebreaker because the research cohorts stamp every epoch with one
  `REFERENCE_DATE`. Guards: `test_sessions_come_back_in_chronological_order`,
  `test_epoch_sessions_sharing_a_timestamp_keep_their_id_order`.
- **`SessionRTMS.demarrer()` takes an optional `date`.** It used to stamp
  `datetime.now()` unconditionally, which silently overwrote the date the caller
  had just passed to the constructor — the sequential seeder's ten sessions,
  spread over 20 days, all landed within one millisecond of each other. The
  follow-up page plots those dates. Replaying a historical course must pass the
  date; only a genuinely new session should default to now.
- **The sequential cohort is 1-indexed (`-S01`…`-S10`) and re-seeding replaces.**
  It used to number from `S00`, so `prochain_index` (count + 1) handed the
  clinical loop `S11` right after `S09`. `prochain_index` now reads the highest
  number already present in the ids, so a deleted visit cannot make the loop
  re-issue an id that `_upsert_session` would then silently overwrite. Seeding
  goes through `Repository.remplacer_sessions_patient`, which drops sessions the
  new cohort no longer has — `sauvegarder_patient` upserts and never deletes
  (correct for the loop, wrong for a seeder), so the old `S00` rows survived the
  renumbering as orphans and sorted into the middle of the course.
- **Cumulating sessions measurably helps — on the sequential cohort only.**
  `python -m src.reporting.sequence_sweep` reruns the same patient-wise CV on the
  same cohort truncated to the first *k* visits and writes
  `data/models/sequence_sweep.json`; page 7 renders it **below** the 2×2, never
  inside it. Measured: AUC **0.923 → 0.996** and accuracy 0.83 → 0.98 from k=1 to
  k=10, with the fold std collapsing **0.109 → 0.008**. The stabilisation is the
  stronger half of the result. Caveat that must travel with the number: this
  simulator injects an alpha biomarker, so 0.92 at a *single* session is already
  inflated — read the curve's shape, not its height. Do **not** append this to
  `comparison.json`: the 2×2's sequence axis is epochs of one recording, so a
  fifth bar in that chart would be an incomparable number.
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

## The article-aligned arm (Arteaga et al., PMC12981298)

The reference study the TDBRAIN track was built from predicts a **continuous**
BDI-II change, fits **one model per rTMS protocol**, and scores with Pearson r
against a permutation null. `src/models/train_article.py` reproduces that setup;
`src/models/train_all.py` still owns the original binary 2×2 and its
`comparison.json`. Keep them separate — r and AUC are not comparable.

Facts worth not re-deriving:

- **`delta_bdi` is confounded, and the confound is bigger than the article's
  result.** Baseline severity alone reaches **r = 0.500** on protocol 1 (0.360 on
  protocol 2, 0.405 pooled) — the same magnitude as the study's headline
  r = 0.401 from EEG. Point estimate above it, but the confound's own
  bootstrap CI is [0.258, 0.700] on 44 patients, which contains 0.401: the two
  are **indistinguishable**, and that is the defensible claim. A model that
  cannot be told apart from a variable requiring no model has not shown added
  value. Measured by `src.reporting.r_stability`.
  You cannot recover 40 points from a BDI of 20. Every regression row therefore
  ships its clinical-only twin (`*_clin_reg`) and a `baseline_r_bdi_pre` column;
  `beats_baseline` is the only verdict that means anything. `pct_reduction` is
  far cleaner (r = 0.156 / −0.054) and is the reason both targets exist in
  `modalities.TARGETS`. Pinned by `test_baseline_severity_alone_predicts_the_target`.
- **Do not score "signal beyond baseline" as a ratio.** Dividing truth *and*
  prediction by `bdi_pre` and correlating the results manufactures correlation
  through the shared divisor: a model emitting a **constant** scored **0.539**
  that way. `metrics.partial_correlation` residualises both sides on the
  covariate instead and correctly returns ~0. `test_the_ratio_trap_is_not_reintroduced`
  keeps the wrong version from coming back as a simplification.
- **Measured result: nothing beats the baseline, nothing is significant.**
  The current `article_comparison.json` (10-fold x 10 repeats, post-leak-fix)
  gives r_oof between −0.192 and −0.419 on the real cohort, every p_perm ≥ 0.86,
  against baselines of 0.500 (P1) and 0.360 (P2). Every R² is negative — the
  models do worse than predicting the cohort mean. The matched simulated cohort
  behaves identically, which is what a negative control should do. This is
  consistent with the binary 2×2's result, not a new failure.
  **Do not quote any single r from this arm as the project's number.** The
  ladder's regression rows, differing only in fold count and standardisation,
  give +0.096 for the same P1 EEG-only cell. `src.reporting.r_stability`
  quantifies that instability directly; see the head-to-head section below.
- **`out_of_fold` returns predictions in original patient order, always.** It used
  to return *fold* order for `repeats=1` and patient order otherwise. Callers line
  these up against per-patient covariates (`bdi_pre`, `age`) held in the dataset's
  order, so fold order silently correlated mismatched vectors — it reported the
  protocol-1 baseline as r = 0.090 instead of 0.500. With repeats, a patient's
  predictions are **averaged** across repeats rather than pooled, so n stays
  n_patients; pooling would inflate every interval and the permutation test.
- **Cohort reconciliation (measured).** `participants.tsv` in the rTMS package has
  exactly 132 rows, all with a protocol and BDI. Protocol 1 is **44 in both** this
  project and the article. Protocol 2 is **88 here vs their 73**. It is *not* an
  indication filter: restricting to `indication == "MDD"` gives 42 + 86, which
  breaks the protocol-1 match. The 15 are therefore excluded by EEG-level quality
  control the article does not enumerate per subject, and **cannot be identified
  from the published metadata**. We keep all 132 and say so — do not drop patients
  to make a number match.
- **The two heads refuse each other.** `LSTMConfig.task` selects a linear head +
  MSE or a logit head + BCE; `predict_proba`/`predict_tri` raise on a regression
  model and `predict_value`/`predict_value_sequence` raise on a classification
  one. This is not defensive noise: BDI-II *points* through a sigmoid render as a
  perfectly plausible probability curve on every page, so it has to fail loudly.
- **The app filters patients by protocol.** `list_patient_ids(repo, protocol)`
  restricts to one arm whenever the selected variant declares one (44 / 88 / 132).
  Without it a protocol-1 checkpoint would be offered protocol-2 patients and
  predict across treatments — the feature vector is the right shape either way,
  so nothing downstream would notice.
- **`app/inference.responder_scale`** is the single conversion putting both heads
  on [0, 1] against the 50 % criterion (regression divides points by the patient's
  own baseline). Suivi and the clinical loop both call it, for the same reason
  `build_model_input` exists: two pages must not disagree about one model's output.
- **`available_models`' `protocol` argument uses a sentinel, not `None`.** `None`
  is a legitimate value meaning *both arms pooled*, so it cannot also mean *not
  supplied*. It did once, and the consequence was invisible: every page reaches
  the registry through `model_choice()` without passing the argument, so
  selecting "Protocole 1" filtered the patient list to the right 44 patients and
  then scored them with the **pooled** checkpoint. Nothing on screen looked
  wrong. Pinned by `test_model_choice_follows_the_selected_protocol`.
- **`recommandation()` takes a `libelle`.** The clinical loop feeds it either a
  probability or a predicted BDI-II reduction — both on [0, 1] against the 50 %
  criterion, which is what lets one function serve both. Hard-coding "P(réponse)"
  told the clinician the model was 29 % *confident* when it was predicting a 29 %
  *improvement*.
- Repeated CV (`cross_validate(..., repeats=n)`) shuffles group-to-fold assignment
  per repeat, as the article's 10×10-fold does. `repeats=1` keeps the **unshuffled**
  splitter so every previously recorded figure reproduces exactly.

- **THE LEAK: early stopping must never watch the outer fold.** `cross_validate`
  used to pass the outer held-out fold straight to `train_one_fold` as its
  early-stopping set, so each fold's checkpoint was *selected* on the very
  patients it was then scored on. `_inner_split` now carves a patient-wise
  early-stopping set out of the **training** fold instead.
  This was not a small bias. With this cohort's near-constant regressor it tuned
  the emitted constant toward each held-out fold's own mean, and the pooled
  out-of-fold r reached **0.61 on shuffled labels** — briefly reading as though
  the project had beaten the reference study's r = 0.401. After the fix, real
  labels give −0.373 and shuffles give −0.18 / +0.24 / −0.49: indistinguishable.
  Guards: `test_early_stopping_never_watches_the_outer_fold`,
  `test_shuffled_labels_do_not_produce_a_correlation`.
- **A permutation test is not a leak test.** `permutation_p` shuffles labels
  *after* prediction, so it only checks the statistic; it reported p = 0.010 for
  the leaked numbers. The decisive check retrains the whole pipeline on permuted
  targets. Any new claim of signal in this project must clear that, not the p.
- **`r_mean` (per-fold) and `r_oof` (pooled) can disagree wildly, and the gap is
  the diagnostic.** The leaked run showed fold-mean **+0.061 ± 0.525** against a
  pooled **+0.606**. With 10-fold on 44 patients a fold holds ~4 patients, so a
  per-fold Pearson r is close to meaningless — but a large mean/pooled gap is a
  reliable signal that something is wrong. Both are recorded in
  `article_comparison.json` for exactly that reason.
- **Predictions are near-constant, and that is the real finding.** `pred_sd` is
  0.1–0.9 BDI-II points against a target sd of 12.75, and every R² is negative.
  The models emit the cohort mean. A positive r alongside a negative R² is a
  partial ranking, never a prediction — do not report the r without the R².
- **Preprocessing now follows the article's Methods** (`TDBRAINConfig`):
  50 Hz notch, **0.01–50 Hz** band-pass (was 1–45), and a **common average
  reference** (`reference="average"`, previously absent entirely). Band power is
  reference-dependent, so the old features were simply not the ones the study
  computed. `recherche_tdbrain.sqlite3` must be re-seeded after changing any of
  these. Still missing versus the article: bad-segment rejection, bad-channel
  interpolation and ICA.
- **The article's model receives no clinical variables**, which is why
  `*_eeg_reg` exists. The multimodal variant takes `bdi_pre` as an **input**, so
  its r cannot separate "the EEG predicted this" from "the intake form did".
  The comparison that means something is **EEG-only vs clinical-only on the same
  arm**; that pairing is what `ARTICLE_ORDERED` orders the table by.
- **The app serves all three cohorts and three axes.** Sidebar =
  **Cohorte → Objectif → (Protocole rTMS, régression seulement) → Jeu de
  variables**. `VISIBLE` exposes the sequential cohort (multi-session loop), the
  matched simulated cohort and TDBRAIN — the last two being the two halves of
  the 2x2.
  This replaces an earlier arrangement whose two justifications both expired:
  the matched cohort was hidden as "a third cohort with neither role" (it *is*
  the simulated half of the 2x2), and there was no "Objectif" radio because
  "each cohort has exactly one head" (TDBRAIN holds **both** — the pooled
  classification pair *and* the six per-protocol regressions). The consequence
  was that **all four 2x2 checkpoints were trained, sidecar'd and unreachable**:
  they appeared only as recorded numbers on page 7. Guard:
  `test_every_2x2_variant_is_reachable_from_the_sidebar`.
- **`available_heads()` reads the disk, not the registry.** A head whose
  checkpoints were never trained must never appear in the sidebar — that was the
  original, still-valid reason the head was not a user choice. It is now a
  per-cohort *set* rather than a constant. Regression sorts first when a cohort
  has one, so every previously recorded default resolves to the same checkpoint.
- **`current_protocol()` returns `None` whenever the classification head is
  selected, and that is load-bearing.** The 2x2 checkpoints are pooled
  (`protocol=None`). Returning an arm under the binary head made
  `available_models` match nothing, so `model_choice` fell back to
  `cfg.models[0]` — serving a checkpoint the sidebar never selected, against a
  patient list filtered to one arm, with nothing on screen looking wrong. Same
  silent-failure shape as the `_UNSET` sentinel. Guard:
  `test_the_binary_head_pools_both_arms`.
- **The 2x2's recorded metrics predate both fixes.** `comparison.json` was
  produced with the old preprocessing *and* the early-stopping leak. Its
  conclusion (everything at base rate) is unlikely to move — a leak that flatters
  a constant regressor does little for AUC — but the numbers are not directly
  comparable to anything trained since. Re-run `train_all` before quoting them
  against the article arm.

## The hypothesis ladder (`new_docs/`, HYPO1–HYPO4 + RES0_AR1 §5)

Two documents in `new_docs/` propose a multi-scale "physics-informed" framework:
Maxwell/Biot–Savart → Hodgkin-Huxley → wave equation → Kuramoto → PLV/coherence
→ Bernoulli hemodynamics → LSTM, plus a Kalman/MPC closed loop, plus an
exploratory Tesla 3-6-9 harmonic hypothesis. `src/reporting/hypo_ablation.py`
executes the part that is testable on TDBRAIN and writes
`data/models/hypo_ablation.json`; page 7 renders it under the article arm.

    python -m src.reporting.hypo_ablation --regression

**What was implemented, and why only that.** Sorted by whether the data can
support it:

- **Implementable and implemented** (`src/preprocessing/connectivity.py`, three
  new modalities `sync` / `cplx` / `h369`): PLV, magnitude-squared coherence,
  the Kuramoto order parameter *and its metastability*, spectral entropy, the
  1/f aperiodic exponent, frontal alpha asymmetry, individual alpha frequency,
  R₃₆₉. 40 columns total.
- **Wilson–Cowan `[E(t), I(t)]` is present only as a proxy.** Fitting the two
  populations to a resting recording is an unidentifiable estimation problem on
  this cohort. The 1/f slope is an accepted proxy for their *ratio* (Gao,
  Peterson & Voytek 2017), and the ratio is what the equations use. Do not
  "upgrade" this to a real Wilson–Cowan fit without new data.
- **Hodgkin–Huxley, the wave equation and the Bernoulli hemodynamic block are
  not implementable.** No membrane recordings, no cortical propagation
  measurement, no perfusion/fNIRS channel exists in TDBRAIN.
- **Model E (the B/E/J electromagnetic layer) is provably empty, not merely
  unimplemented.** Coil current, coil geometry and tissue conductivity are all
  unpublished, so every derivable physical quantity collapses to a function of
  the protocol integer, which takes two values. `physics_is_collinear()` proves
  it by rank: adding three physics columns to `[1 | protocol]` leaves the rank
  at 2. Pinned by `test_physics_proxy_adds_no_rank_over_the_protocol`.
- **Kalman/MPC closed loop is a design, not a measurement.** There is no
  closed-loop data anywhere in this project and the dose is unpublished, so
  `J(u)` has nothing to minimise over. `recommandation()` already reports
  direction only, for the same reason.

**Measured result — the equations do not improve prediction, on any target.**

- Binary responder, 132 patients, pooled, patient-wise 5-fold: **all eleven
  rungs at chance.** AUC 0.384–0.491, every 95% CI straddling 0.5, every PR-AUC
  *below* the 0.629 base rate, every Brier at or above the 0.233 no-skill value.
  The new families are the best of the lot (`sync` 0.491, `cplx` 0.489) and that
  is still chance.
- Continuous target, per protocol: every R² negative, nothing clears its
  clinical baseline on `delta_bdi` (0.500 on P1, 0.360 on P2), nothing reaches
  the article's r = 0.401. The best network arm is P1 `NET` at r = +0.171
  (p = 0.158).
- **Shuffled-label control on the best rung: 0.441 / 0.512 / 0.448 against a
  real 0.491.** Indistinguishable. This is the decisive test, not the
  permutation p.
- **Univariate screen: 4/40, 4/40 and 7/40 nominal hits across the three
  targets, and zero survive Benjamini–Hochberg.** ~2 nominal hits are the
  expectation under the null at 40 tests.

**The positive control is what makes the null result worth reporting.** The same
features reproduce three replicated age effects with the right sign:
aperiodic exponent r = −0.335 (p = 8.7e-5, 1/f flattens with age), individual
alpha frequency r = −0.174 (p = 0.046), alpha-band PLV r = −0.295 (p = 6.0e-4).
Mean aperiodic exponent 1.35 and mean IAF 9.44 Hz are both textbook values. The
instrument works; the task is the problem. Without this control, "connectivity
predicts nothing" and "we computed connectivity wrongly" produce identical
tables.

**H369 is refuted by its own criterion.** The document requires it be declared
"non soutenue" if the block fails out-of-sample or dies under multiple-comparison
correction. Both happened: G → H369 moves AUC by +0.001, and `r369_f0_1` — which
was the single *best* feature for `delta_bdi` at r = +0.239, p = 0.006 — has
q = 0.235. That nominal p is exactly the kind of hit that becomes a paper when
reported uncorrected. Note it is *not* a restatement of band power: on real data
R₃₆₉ correlates with relative alpha power at only r ≈ 0.05.

Other facts worth not re-deriving:

- **The new blocks are deliberately tiny (30 / 6 / 4).** A per-pair connectivity
  block would be 325 × 5 = 1625 columns on 132 patients, and this project has
  already measured that 139 features lose to 4. Aggregation over anatomically
  motivated groups (frontal = the rTMS target, left/right = the two protocols)
  happens before the model ever sees them.
- **`MODALITY_ORDER` grew by appending, never by inserting.** The new blocks sit
  *after* `ecg` even though they belong next to `eeg` thematically, because every
  existing checkpoint must keep receiving a byte-identical vector. Pinned by
  `test_new_blocks_are_appended_so_old_checkpoints_keep_their_vector`.
- **`cross_validate(standardise=True)` is cohort-level and fold-safe, and is not
  `zscore_epochs`.** It fits mean/std on the *training fold only* and preserves
  between-patient variance; per-patient z-scoring removes exactly the quantity a
  between-patient label lives in. It defaults to **off** so every previously
  recorded figure still reproduces byte-for-byte. The network blocks need it
  because they mix PLV (~0.2), SDNN (~50 ms) and alpha peak (~10 Hz).
- **Connectivity refuses a single-channel dataset rather than falling back.**
  PLV between one channel and itself is 1.0 and would pass every shape
  assertion. `eeg_block`'s `signals[:, :, None, :]` fallback is wrong here.
- **`beats_chance()` is the stopping rule, and accuracy is deliberately not in
  it.** All four of: AUC CI excluding 0.5, permutation p ≤ α, balanced accuracy
  > 0.5, and both classes actually predicted. On this cohort the all-positive
  predictor scores accuracy 0.629 and F1 0.768 — which is how a null result gets
  published as a positive one.
- **`benjamini_hochberg` is hand-written, not statsmodels.** Six lines, and
  every dependency here needs an upper bound and a justification.

### Head-to-head with the article: what a single *r* is worth here

`src/reporting/r_stability.py` retrains the **exact** configuration the study
reports — protocol 1, EEG only, continuous `delta_bdi` — fifteen times on real
labels and fifteen times on permuted ones. Measured:

| | protocol 1 (n=44) | protocol 2 (n=88) |
|---|---|---|
| our r, real labels | −0.010 ± 0.123, range [−0.199, +0.182] | −0.157 ± 0.143 |
| our r, permuted labels | +0.031 ± 0.125, range [−0.154, +0.229] | −0.085 ± 0.113 |
| overlap (shuffled runs ≥ mean real r) | 60 % | 67 % |
| max \|r\| reachable with **zero** label information | 0.229 | 0.231 |
| trivial baseline `bdi_pre` alone | 0.500, CI [0.258, 0.700] | 0.360, CI [0.147, 0.562] |
| the article | 0.401 | 0.26 |

Four things follow, and they must travel together:

- **Our EEG r is indistinguishable from its own null.** Real and shuffled
  distributions sit on top of each other (60 % / 67 % overlap). We do **not**
  reproduce r = 0.401, and saying so is not a hedge.
- **The article's number is *not* explainable as noise at this n.** 0.401 exceeds
  the largest \|r\| our pipeline reaches on randomised labels (0.229). Their
  extraction — ICA, itEMD, SBLEST spatial filters — is doing something ours
  (per-channel band-power averages) does not. That is the honest reading of the
  gap, and it is a gap in *our* feature extraction.
- **But 0.401 is not distinguishable from the confound either.** `bdi_pre` alone
  reaches 0.500 with CI [0.258, 0.700], which contains 0.401; 80 % of bootstrap
  resamples put the confound higher. So the defensible claim is **not** "we beat
  them" and **not** "0.500 > 0.401" — it is that their EEG model cannot be told
  apart from a variable that needs no model and no EEG, and they do not report
  that comparison. On `pct_reduction`, which divides the coupling out, the
  confound drops to r = 0.156 (p = 0.313).
- **`article_comparison.json` and the ladder's regression rows disagree in
  sign** (−0.373 vs +0.096 for P1 EEG-only) because they differ in fold count
  and standardisation. That disagreement is not a bug to reconcile: it is the
  same instability the table above quantifies. Do not quote either as *the*
  number. CLAUDE.md previously carried "best real arm r = +0.163" from a
  superseded run; the current file says −0.215 for that row.

## Conventions

- **Language**: French domain terms (`ajouter_dossier`, `frequence_hz`,
  `localisation`) because the UML diagram uses French. The jury looks for these.
  Python identifiers use snake_case (no accents).
- **Supported Python is 3.11–3.14, and `requirements.txt` enforces it.** The
  first two lines are unsatisfiable requirements guarded by `python_version`
  markers, so an unsupported interpreter fails immediately with a readable name
  (`NeuroReponse-requires-Python-3.11-to-3.14-yours-is-too-new`). Without them,
  Python 3.15 gets three packages in and dies inside a Meson source build
  ("Could not find vswhere.exe") because pandas, torch, scikit-learn, matplotlib
  and pyarrow ship **no cp315 wheels** (verified against PyPI, 2026-08). numpy,
  scipy and pillow do — which is why the failure looks like a pandas bug and
  isn't. Do not "fix" it by installing Visual Studio Build Tools: torch cannot
  realistically be built from source. Raise the bound when upstream ships 3.15
  wheels; `tests/test_requirements.py` keeps the guard first in the file (pip
  collects in file order, so position is what makes it short-circuit) and keeps
  the README's stated range in sync.
- **Every dependency carries an upper bound**, one major above the tested
  version — enforced by `test_every_dependency_has_an_upper_bound`. pandas 3.0
  landed silently under the old `pandas>=2.1`; the next major would too.
- **ML backend**: **PyTorch**, not TensorFlow — TF doesn't ship wheels for
  Python 3.14 (cp310–cp313 only). Architecture and metrics are identical to what
  the doc describes.
- **Persistence**: SQLAlchemy 2.0 ORM in `src/db/schema.py`; all DB access goes
  through `Repository` (= the `BaseDeDonnées` UML class).
- **Cross-validation**: always patient-wise (`GroupKFold` with patient id as
  group). The test `test_groupkfold_keeps_patients_separated` enforces this.
- **Normalization**: per-patient, never across patients. The test
  `test_pipeline_no_cross_patient_leakage` enforces this.

## Testing

- `pytest tests/ -q` should always pass (332 tests).
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

python -m src.models.train_article --real-only --n-splits 10 --repeats 10  # article arm (6 checkpoints)
python -m src.reporting.figures              # regenerate README figures
python -m src.reporting.hypo_ablation --regression   # new_docs hypothesis ladder
python -m src.reporting.r_stability          # how much is one Pearson r worth?
streamlit run src/app/main.py                # launch app
jupyter nbconvert --to notebook --execute notebooks/03_results_analysis.ipynb \
    --output 03_results_analysis.ipynb       # re-run results notebook
```
