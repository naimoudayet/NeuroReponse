# Real EEG data: the TDBRAIN rTMS-in-MDD cohort

This project's default dataset is **simulated** (`src/data/simulator.py`). To let the
model learn from **real** EEG, `src/data/tdbrain.py` loads the public **TDBRAIN**
database — specifically the subset of major-depression patients treated with rTMS,
who have pre-/post-treatment BDI scores. This is the on-target real dataset behind
the reference study *"Multiband EEG signature decoded using machine learning for
predicting rTMS treatment response in major depression"* ([PMC12981298](https://pmc.ncbi.nlm.nih.gov/articles/PMC12981298/)).

No patient data lives in this repo (see CLAUDE.md). You download TDBRAIN yourself and
point the loader at your local copy.

## What TDBRAIN gives you (and what it does not)

| | |
|---|---|
| Modalities | **EEG + ECG.** Every recording carries an ECG lead (`Erbs`, Erb's point) alongside the 26-channel montage, so `ecg` is populated with an RR tachogram. **No ERP**: `erp` stays `None` — see the modality matrix below. |
| Time axis | **One pre-treatment resting recording per patient.** There is *no* rTMS-session trajectory — the LSTM's sequence axis is filled with **epochs** of the ~2-min recording. |
| Label | **Binary responder** = ≥50% BDI-II reduction pre→post. Raw `delta_bdi` and `pct_reduction` are kept in `metadata` for an optional regression later. |
| Cohort | MDD patients across rTMS **protocol 1** (10 Hz L-DLPFC) and **protocol 2** (1 Hz R-DLPFC). Evaluate them **separately** — they're different treatments. |
| Condition | Eyes-open (`EO`) is the default; the reference study found eyes-closed non-significant. |
| Channels | 26-channel 10-10 montage, 500 Hz native (downsampled to 250 Hz), plus 4 EOG, 1 ECG, 1 EMG and a trigger channel (33 in the BDF). |

### Which modalities co-occur (measured on TDBRAIN V3.1)

| Combination | Subjects |
|---|---|
| EEG | 1300 on disk |
| EEG + ECG | all of them — the `Erbs` lead is in every recording |
| **rTMS outcome + EEG + ECG** | **190** (163 with `formal_status == MDD`) |
| rTMS outcome + EEG + ECG + **ERP** | **0** |

The oddball/ERP task exists (129 subjects, with a full `events.tsv`) but **every one
of them is a healthy control** — no BDI, no rTMS protocol. ERP and treatment outcome
are perfectly disjoint in TDBRAIN, so a four-modality model is not merely unpopulated
here, it is impossible. The ERP arm of the NPDT design stays on the simulated track.

**rTMS is outcome-only.** The dataset records `rTMS PROTOCOL` (1 or 2), `Responder`,
`Remitter` and BDI pre/post — but no stimulation intensity, motor threshold, train
duration/count, or number of sessions. The three protocol-derived fields
(`frequence_hz`, `localisation`, `protocole`) are all deterministic functions of the
protocol integer, so they carry **one bit**, and that bit is empirically null for
response: 61.4% responders under protocol 1 vs 64.4% under protocol 2
(χ² p = 0.885, AUC 0.514, mutual information 0.0004 nats). It is also confounded —
protocol-2 patients are ~6 years older (p = 0.013). Use `protocols=(1,)` / `(2,)` to
**stratify**, never as a feature.

Because we use one baseline recording, expect **lower performance than a per-session
model would suggest** — and, with a single channel or without spatial filtering, lower
than the reference paper's spatial-filter method (they reported Pearson r ≈ 0.40 / 0.26
by protocol on the *continuous* ΔBDI). This loader gives the model the full montage
(`tdbrain_features`) to give it a fair chance.

## How to get the data

1. Go to **https://brainclinics.com/resources/** and open the TDBRAIN dataset.
2. Register / sign in (ORCID) and accept the **data-use agreement**. Access is gated;
   the public release is adults (≥18) who consented to data sharing.
3. Download and unzip. You want the folder that contains **`participants.tsv`** and a
   **`derivatives/`** tree of per-subject EEG CSVs.

> **Licensing:** TDBRAIN's data-use agreement governs redistribution and derived data.
> Do not commit any of it (raw or processed) to this repo. Keep it outside the working
> tree or in a git-ignored path.

## Expected on-disk layout

```
<root>/
  participants.tsv                     # one row per subject: id, indication, BDI pre/post, rTMS protocol, ...
  dataset_description.json
  sub-XXXXXXXX/                        # subjects sit at the root in V3.1
    ses-1/
      eeg/
        sub-XXXXXXXX_ses-1_task-restEO_eeg.bdf       # BioSemi BDF (real TDBRAIN format)
        sub-XXXXXXXX_ses-1_task-restEO_eeg.json      # SamplingFrequency, reference, ...
        sub-XXXXXXXX_ses-1_task-restEO_channels.tsv  # name / type / units per channel
        sub-XXXXXXXX_ses-1_task-restEC_eeg.bdf
```

The file finder is a recursive glob, so a `derivatives/`-nested layout (used by the
synthetic fixture and by some derivative exports) works equally well.

The real download ships EEG as **BioSemi BDF** (`*_eeg.bdf`, 500 Hz, 26 EEG +
EOG/ECG/EMG channels). The loader reads these with [`mne`](https://mne.tools)
(`mne.io.read_raw_bdf`) — `mne` is a required dependency (see `requirements.txt`);
its BDF reader is self-contained (`edfio` is only needed to *write* BDF for the
test fixture). The file finder globs for the subject id + `EO`/`EC`, preferring
`.bdf` and falling back to `.csv` (the synthetic fixture and any derivative CSV
exports; the CSV reader prefers a channel-named header row, else infers orientation).

**Preprocessing.** Raw BDF is unfiltered — it carries a large DC offset and heavy
power-line noise. On read, the loader applies a configurable power-line notch
(`TDBRAINConfig.notch_hz`, default 50 Hz + harmonics) and band-pass
(`TDBRAINConfig.bandpass_hz`, default 1–45 Hz). Set either to `None` to disable
(e.g. if you point the loader at already-cleaned derivative files). This is *not*
the reference study's full artifact-removal (ICA) pipeline — it is the minimum to
get physiologically meaningful band powers.

**The autonomic channel.** `TDBRAINConfig.ecg_channel` (default `"Erbs"`) enables
R-peak detection on the ECG lead, from the *same* BDF read as the EEG. Detection
runs at the native 500 Hz — before the EEG notch and the downsample to 250 Hz,
both of which blunt R-peak timing. The detector is a compact Pan–Tompkins
(band-pass 5–20 Hz → derivative → square → 150 ms integrate → peak-pick), followed
by two rejection stages: a physiological gate (30–200 bpm) and a Malik
successive-difference filter that drops beats deviating >20% from local context.
Both matter: a plain `|x| > 3σ` threshold mistakes the T wave for a second R peak
and inflates RMSSD roughly fourfold, and unfiltered missed beats inflate SDNN by an
order of magnitude. Set `ecg_channel=None` to skip it.

Validated on the real cohort: heart rates land in 55–88 bpm with SDNN 15–74 ms
across every patient sampled. Patients whose lead is unusable keep their EEG and
get a zero tachogram (neutral HRV), rather than being dropped.

**HRV is a patient-level trait here, not a time series.** An 8 s epoch holds only
~9 beats — far too few for SDNN, and below the 16-sample floor `hrv_features` needs
for LF/HF. So the tachogram is measured once over the full ~120 s (~140 beats) and
repeated along the epoch axis, exactly like the BDI scores. The direct consequence:
**the HRV block must never go through per-patient z-scoring.** It has zero
within-patient variance, so z-scoring divides by a zero std and collapses the whole
block to constant 0 — the modality silently disappears while every shape assertion
still passes. `tdbrain_features` z-scores the EEG block only; the regression guard
is `test_zscoring_does_not_blank_the_hrv_block`.

## Column mapping

`participants.tsv` column names are auto-resolved case-insensitively. If your download
uses different headers, set them explicitly on `TDBRAINConfig`:

| Meaning | Config field | Auto-detected candidates |
|---|---|---|
| Subject id | `col_id` | `participants_ID`, `participant_id`, `id` |
| Diagnosis | `col_indication` | `indication`, `diagnosis`, `dx` |
| Baseline BDI | `col_bdi_pre` | `BDI_pre`, `BDI_baseline`, `BDI_t0`, … |
| Post BDI | `col_bdi_post` | `BDI_post`, `BDI_end`, `BDI_t1`, … |
| rTMS protocol | `col_protocol` | `rTMS_protocol`, `protocol`, … |

## Usage

```python
from src.data.tdbrain import TDBRAINConfig, load_tdbrain, tdbrain_features
from src.models.train import cross_validate

cfg = TDBRAINConfig(root="/path/to/tdbrain")     # EO, protocols 1&2, 8 epochs of 8 s
ds = load_tdbrain(cfg)                            # -> LoadedDataset (real EEG + ECG, responder labels)

x, y, groups, names = tdbrain_features(ds)        # (n_patients, n_epochs, 26*5 band powers)
result = cross_validate(x, y, groups)             # patient-wise GroupKFold, same as simulated
print(result.summary())
```

Add the autonomic modality (130 band powers + 5 HRV = 135 features):

```python
x, y, groups, names = tdbrain_features(ds, modalities=("eeg", "ecg"))
assert names[-5:] == ("hrv_hr_mean", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_lf_hf")
```

Evaluate one protocol at a time:

```python
p1 = load_tdbrain(TDBRAINConfig(root="/path/to/tdbrain", protocols=(1,)))
p2 = load_tdbrain(TDBRAINConfig(root="/path/to/tdbrain", protocols=(2,)))
```

Baseline (non-temporal) classifier instead of the epoch-LSTM:

```python
ds = load_tdbrain(TDBRAINConfig(root="/path/to/tdbrain", snapshot=True))  # one window/patient
```

## Using the cohort in the Streamlit app

The app has a **data-source selector** in the sidebar (`Données simulées` /
`TDBRAIN (EEG réel)`). Each source has its own SQLite file and its own checkpoint,
so real and simulated patients never mix:

```powershell
# 1. seed the real cohort into its own database (reads one BDF per patient)
python -m src.data.tdbrain_seeder `
    --root "data/tdbrain/TDBRAIN_Dataset_V3_1_Encr/TDBRAIN_Dataset_V3_1" `
    --db recherche_tdbrain.sqlite3

# 2. train + persist a model with its feature-contract sidecar
python -m src.models.train_tdbrain `
    --root "data/tdbrain/TDBRAIN_Dataset_V3_1_Encr/TDBRAIN_Dataset_V3_1"

# (or do both from a single load, avoiding a second BDF pass)
python -m src.models.train_tdbrain --root <root> --seed-db recherche_tdbrain.sqlite3

streamlit run src/app/main.py
```

**What the app stores.** One `SessionRTMS` per *epoch* (8 by default), one
`SignalNeurophysiologique` per *channel* (26 EEG) **plus one ECG signal** holding
that epoch's RR tachogram — roughly 220 MB of BLOBs for 132 patients. The ECG row
is stored with `sampling_rate_hz = 0.0` on purpose: an RR series is event-sampled,
not uniformly sampled, so any Hz value would be a lie the UI might plot as a time
axis. `SignalNeurophysiologique.extraire_features()` dispatches on modality and
returns HRV metrics for ECG. The Sessions page is relabelled "Époques du repos"
under TDBRAIN, plots the tachogram against beat index, and states that rTMS
parameters and BDI scores are identical across epochs, because a single recording
carries no treatment trajectory.

**Follow-up view.** The **Suivi** page synthesises a patient's whole course from
every session: clinical trajectory, model (TRI) trajectory, and whether the two
agree. Under TDBRAIN it detects that all epochs carry identical BDI-II scores and
says so explicitly — no trend is fitted, and the TRI spread is reported as
*coherence between windows of one recording*, not as progress. The analysis lives
in `src/reporting/suivi.py` so it is unit-tested independently of Streamlit.

**Feature contract.** The simulated model eats 8 features from one channel; the
TDBRAIN model eats 130 (26 channels × 5 bands), or **135** with the ECG modality.
Each TDBRAIN checkpoint is saved with a JSON sidecar
(`data/models/tdbrain_response_v1.json`) recording `fs`, `channels`, `n_epochs`,
`modalities`, `input_size` and whether per-patient z-scoring was applied. The
Predictions page rebuilds its inputs from that sidecar — selecting channels **by
name**, never by database row order — and refuses to predict if the stored data
does not match, including when a model trained with HRV meets a patient seeded
without an ECG. A final width check against `input_size` catches anything else.

**Measured result (132 patients, 5-fold patient-wise, all four variants).**

| Modalities | Normalisation | AUC | Accuracy | F1 |
|---|---|---|---|---|
| EEG | raw | 0.465 ± 0.134 | 0.628 | 0.768 |
| EEG | per-patient z | 0.467 ± 0.103 | 0.628 | 0.768 |
| EEG + ECG | raw | **0.516 ± 0.078** | 0.628 | 0.768 |
| EEG + ECG | per-patient z | 0.500 ± 0.093 | 0.628 | 0.768 |

Adding HRV moves AUC by **+0.050**, from below chance to barely above it. **Do not
read that as the autonomic channel working.** Two things say otherwise:

1. The fold-to-fold standard deviation (±0.078) is larger than the gain.
2. **Accuracy and F1 are byte-identical across all four variants** — 0.628 is
   exactly the responder base rate (83/132), and 0.768 is the F1 of an all-positive
   predictor. The model is predicting "responder" for every patient in every
   configuration. It has learned nothing; the AUC is noise around 0.5.

So the conclusion in CLAUDE.md stands and is now better supported: single-baseline
rTMS response prediction is a **known-hard negative result**. What the ECG track
buys is not accuracy but *evidence* — the multimodal claim is now measured on real
data with a real ablation, rather than asserted from the simulator. Reporting a
null result with a clean protocol is a defensible outcome; reporting +0.050 as a
win would not be.

## Verify against your download before trusting results

The parser is built from the TDBRAIN data descriptor ([PMC9198070](https://pmc.ncbi.nlm.nih.gov/articles/PMC9198070/)),
not from the gated files. When you first run it, confirm:

- [ ] `participants.tsv` columns resolved to the right fields (print `metadata.head()`).
- [ ] Responder counts and `pct_reduction` look sane for your cohort.
- [ ] EEG BDF files load with the expected channel names (no "channels missing"
      error). The 26-channel montage must all be present; adjust `TDBRAIN_CHANNELS_26`
      / `_CHANNEL_ALIASES` in `src/data/tdbrain.py` if your montage differs. (Headerless
      or transposed CSV exports go through `_read_condition_csv` instead.)
- [ ] The number of loaded patients matches expectation; check the "excluded patients
      by reason" warning to see what was filtered and why.

## Dry run without the real data

```bash
python -m src.data.tdbrain          # writes a synthetic TDBRAIN-format tree and loads it
```

`make_synthetic_tdbrain()` generates a miniature tree in the same format (used by
`tests/test_tdbrain.py`). It is **not** medical data — just enough structure to
exercise the parser and the full load → features → LSTM path.
