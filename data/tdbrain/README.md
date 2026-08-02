# Paste your TDBRAIN download here

This folder is where the **real** TDBRAIN EEG data goes. It is **git-ignored** — nothing
you drop here will be committed (the data is under a Data Use Agreement; see
`docs/tdbrain.md`). Only this README and `.gitkeep` are tracked.

## What to download

From https://brainclinics.com/resources/tdbrain-dataset :

1. **TDBRAIN Treatment Prediction dataset V3.1 & template** ← the one for our task.
2. **TDBRAIN-specific pre-processing Python code** (optional, helps confirm the format).
3. *Only if the Treatment package has no EEG signals:* **TDBRAIN Dataset V3.1** (full).

## Where to paste

Unzip the download and paste its contents **directly into this folder**, so that
`participants.tsv` (or the labels file) sits at the top level here:

```
data/tdbrain/
├── participants.tsv            <- metadata + BDI pre/post + rTMS protocol
├── participants.json
└── derivatives/
    └── sub-XXXXXXXX/
        └── ses-1/
            └── eeg/
                ├── sub-XXXXXXXX_ses-1_task-restEO_eeg.csv
                └── sub-XXXXXXXX_ses-1_task-restEC_eeg.csv
```

The exact layout may differ (the raw archive ships EEG as **BDF/BDF+**, and the
metadata may be an `.xlsx`). That's fine — the loader will be adjusted to match.

## After pasting

1. Send me the folder tree so I can align the loader:
   ```powershell
   Get-ChildItem -Recurse data/tdbrain | Select-Object -First 60 FullName
   ```
2. The notebook `notebooks/04_tdbrain_real_data.ipynb` auto-detects this folder
   (it looks for `data/tdbrain/participants.tsv`). Re-run it top to bottom and it
   switches from the synthetic demo to your real data.

> If you unzip somewhere else, set an environment variable instead of pasting here:
> `setx TDBRAIN_ROOT "D:\path\to\tdbrain"` (then reopen the terminal).
