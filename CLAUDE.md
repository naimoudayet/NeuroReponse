# CLAUDE.md — guidance for future sessions

## Project state

**Phases 1–6 complete.** This is a finished PFE prototype: 32/32 tests pass,
ruff clean, three executable notebooks, working Streamlit app with PDF export.
Source of truth for the design remains `Description et Conception (Version0) (3).docx`.

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

- `pytest tests/ -q` should always pass (32 tests).
- `python -m ruff check src/ --select=E,F,I,UP --ignore=E501` must be clean.
- `python -m jupyter nbconvert --to notebook --execute notebooks/0X_*.ipynb`
  must execute all three notebooks end-to-end without error.

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
  (preprocessing, training, app) will work unchanged.
- **Add EMG/ERP**: same as fNIRS above.

## Useful commands

```powershell
python -m pytest -q                          # tests
python -m ruff check src/                    # lint
python -m src.data.simulator                 # regenerate simulated data
python -m src.data.seeder                    # seed SQLite from simulated data
python -m src.reporting.figures              # regenerate README figures
streamlit run src/app/main.py                # launch app
jupyter nbconvert --to notebook --execute notebooks/03_results_analysis.ipynb \
    --output 03_results_analysis.ipynb       # re-run results notebook
```
