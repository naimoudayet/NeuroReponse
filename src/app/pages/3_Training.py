from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app.utils import (
    PROTOCOLES,
    get_repository,
    has_trained_model,
    list_patient_ids,
    model_choice,
    source_config,
    source_selector,
)
from src.models.lstm import CLASSIFICATION, REGRESSION, LSTMConfig
from src.models.train import TrainConfig, cross_validate, fit_final_model, save_model
from src.models.train_tdbrain import sidecar_path

st.set_page_config(page_title="Training", page_icon=":robot_face:", layout="wide")
st.title("Entraînement du modèle LSTM")

source = source_selector()
cfg_src = source_config(source)
choix = model_choice(source)
is_real = cfg_src.is_real
unit = f"{cfg_src.unit}s"

# The 2x2 variants declare their modalities and read the app's own database; the
# legacy checkpoint predates both and keeps its fixed 8-feature pipeline.
from_db = bool(choix.modalities)
MODEL_PATH = choix.model

st.caption(
    f"**{cfg_src.label}** · modèle **{choix.label}** · cible "
    f"**{'ΔBDI (régression)' if choix.is_regression else 'réponse binaire'}**"
    + (f" · {PROTOCOLES[choix.protocol]}" if choix.protocol is not None else "")
    + f" → `{MODEL_PATH}`"
)
if choix.is_regression:
    st.info(
        "**Arm aligné sur l'article** (Arteaga et al., PMC12981298) : cible "
        "continue, protocoles séparés, score = corrélation de Pearson. "
        "⚠️ `delta_bdi` est couplé à la sévérité initiale — sur le protocole 1, "
        "`bdi_pre` seul atteint r = 0.500. Comparez toujours au modèle "
        "**clinique seul** du même bras avant de conclure quoi que ce soit."
    )


# --------------------------------------------------------------------------- #
# Data loading — simulated from disk, TDBRAIN from the app's own database.
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Lecture de la cohorte…")
def _load_simulated():
    from src.app.utils import SIM_DIR
    from src.data.loader import load
    from src.preprocessing.pipeline import PipelineConfig, preprocess

    ds = load(SIM_DIR)
    pre = preprocess(ds.signals, PipelineConfig(fs=ds.fs, mode="features"))
    groups = np.arange(pre.x.shape[0])
    return pre.x, ds.labels.astype(np.float32), groups, None, float(ds.fs), int(ds.window)


@st.cache_data(show_spinner="Lecture de la cohorte depuis la base…")
def _load_from_db(
    db: str, modalities: tuple[str, ...], zscore: bool,
    target: str = "responder", protocol: int | None = None,
):
    """Rebuild this variant's inputs from the app's own database.

    Deliberately the *same* two calls training makes — ``dataset_from_repository``
    then ``build_features``. Assembling the blocks here instead would let the page
    train a checkpoint on a vector its own Predictions page would never rebuild.
    """
    from src.data.modalities import build_features
    from src.data.tdbrain_seeder import dataset_from_repository
    from src.db import Repository

    ds = dataset_from_repository(Repository(db_url=f"sqlite:///{db}"))
    x, y, groups, names = build_features(
        ds, modalities=modalities, per_patient_zscore=zscore,
        target=target, protocol=protocol,
    )
    return ds, x, y.astype(np.float32), groups, names


if from_db:
    zscore = st.sidebar.checkbox(
        "Normalisation z-score intra-patient", value=True,
        help=(
            "Centre/réduit chaque patient sur ses propres époques — bloc EEG "
            "uniquement. Les blocs clinique et HRV sont constants d'une époque à "
            "l'autre : les normaliser les annulerait."
        ),
    )
    if not list_patient_ids(get_repository(source)):
        commande = (
            "python -m src.data.tdbrain_seeder --root "
            '"data/tdbrain/TDBRAIN_Dataset_V3_1_Encr/TDBRAIN_Dataset_V3_1"'
            if is_real
            else "python -m src.data.tdbrain_seeder --matched"
        )
        st.error(
            f"Base vide pour **{cfg_src.label}**. Charge la cohorte d'abord :\n\n"
            f"```powershell\n{commande} --db {cfg_src.db}\n```"
        )
        st.stop()

    try:
        dataset, x, y, groups, feature_names = _load_from_db(
            str(cfg_src.db), tuple(choix.modalities), zscore,
            choix.target, choix.protocol,
        )
    except (ValueError, KeyError) as exc:
        st.error(f"Impossible de reconstruire les entrées de ce modèle : {exc}")
        st.stop()
    channels, fs, window = list(dataset.channels or []), float(dataset.fs), int(dataset.window)
else:
    from src.app.utils import SIM_DIR

    if not (SIM_DIR / "eeg_simulated.npz").exists():
        st.error(
            "Le dataset simulé n'existe pas encore. Lance d'abord l'initialisation depuis "
            "l'accueil (ou : `python -m src.data.simulator`)."
        )
        st.stop()
    zscore = True
    dataset = None
    x, y, groups, channels, fs, window = _load_simulated()
    feature_names = ()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients", x.shape[0])
c2.metric(f"{unit.capitalize()} / patient", x.shape[1])
c3.metric("Features", x.shape[-1])
c4.metric("Modèle existant", "oui" if has_trained_model(source) else "non")
if from_db:
    blocs = []
    if "rtms" in choix.modalities:
        blocs.append("4 cliniques (protocole, âge, sexe, BDI-II)")
    if "eeg" in choix.modalities:
        blocs.append(f"{len(channels)} canaux × 5 bandes = {len(channels) * 5}")
    if "ecg" in choix.modalities:
        blocs.append("5 HRV (ECG)")
    st.caption(
        f"{int(y.sum())} répondeurs / {len(y)} patients "
        f"(taux de base {max(y.mean(), 1 - y.mean()):.0%}) · "
        f"{' + '.join(blocs)} · {fs:g} Hz."
    )

st.markdown("##### Hyperparamètres")
left, right = st.columns(2)
with left:
    epochs = st.slider("Epochs", 5, 60, 30)
    batch_size = st.select_slider("Batch size", options=[4, 8, 16, 32], value=8)
    lr = st.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                          value=3e-3, format_func=lambda v: f"{v:.0e}")
with right:
    n_splits = st.slider("Folds (CV patient-wise)", 3, 10, 5)
    repeats = st.slider(
        "Répétitions de la CV", 1, 10, 1,
        help="L'article en utilise 10 : sur 44 patients, un seul découpage "
             "dépend beaucoup de quels 9 patients tombent en validation.",
    )
    dropout = st.slider("Dropout", 0.0, 0.6, 0.3, step=0.05)
    seed = st.number_input("Seed", 0, 1000, 0)


def _configs():
    return (
        LSTMConfig(
            input_size=x.shape[-1], hidden_sizes=(128, 64), dropout=dropout,
            task=REGRESSION if choix.is_regression else CLASSIFICATION,
        ),
        TrainConfig(epochs=epochs, batch_size=batch_size, lr=lr,
                    early_stopping_patience=max(3, epochs // 6), seed=seed),
    )


st.divider()
tab_cv, tab_final = st.tabs(["Validation croisée", "Modèle final"])

with tab_cv:
    if st.button("Lancer la validation croisée patient-wise", type="primary"):
        lstm_cfg, train_cfg = _configs()
        with st.spinner("Entraînement en cours…"):
            cv = cross_validate(x, y, groups, lstm_cfg=lstm_cfg,
                                train_cfg=train_cfg, n_splits=n_splits,
                                repeats=repeats)
        st.session_state[f"last_cv_{source.value}:{choix.key}"] = cv

    cv = st.session_state.get(f"last_cv_{source.value}:{choix.key}")
    if cv:
        summary = cv.summary()
        a, b, c = st.columns(3)
        if cv.is_regression:
            from src.models.metrics import regression_report

            a.metric("r (moyenne des plis)", f"{summary['r_mean']:+.3f}",
                     f"± {summary['r_std']:.3f}")
            b.metric("MAE (points BDI-II)", f"{summary['mae_mean']:.2f}")
            c.metric("R²", f"{summary['r2_mean']:+.3f}")
            y_true, y_pred = cv.out_of_fold(y)
            pooled = regression_report(y_true, y_pred)
            st.caption(
                f"Hors-pli groupé : r = {pooled['r']:+.3f}, "
                f"p (permutation) = {pooled['p_perm']:.3f}, n = {pooled['n']}. "
                f"Un R² négatif signifie que le modèle fait pire que prédire la "
                f"moyenne de la cohorte."
            )
            if "bdi_pre" in dataset.metadata:
                from src.data.modalities import protocol_mask
                from src.models.metrics import pearson_r

                mask = protocol_mask(dataset, choix.protocol)
                base = pearson_r(
                    y_true, dataset.metadata["bdi_pre"].to_numpy(float)[mask]
                )
                if pooled["r"] <= base:
                    st.warning(
                        f"**Le BDI-II de référence seul fait mieux** "
                        f"(r = {base:+.3f} contre {pooled['r']:+.3f}). Ce modèle "
                        f"n'apporte rien au-delà du dossier d'admission."
                    )
        else:
            a.metric("Accuracy", f"{summary['accuracy_mean']:.2%}", f"± {summary['accuracy_std']:.2%}")
            b.metric("AUC", f"{summary['auc_mean']:.2%}", f"± {summary['auc_std']:.2%}")
            c.metric("F1", f"{summary['f1_mean']:.2%}", f"± {summary['f1_std']:.2%}")

        if from_db and not cv.is_regression:
            base = max(y.mean(), 1 - y.mean())
            if summary["auc_mean"] < 0.6:
                st.warning(
                    f"AUC ≈ {summary['auc_mean']:.2f} : proche du hasard. La réponse au "
                    f"rTMS est ici prédite à partir d'**un seul enregistrement de repos**, "
                    f"sans trajectoire de traitement — un résultat négatif honnête, "
                    f"cohérent avec la littérature. L'accuracy ({summary['accuracy_mean']:.0%}) "
                    f"reste proche du taux de base ({base:.0%}) : le modèle prédit "
                    f"essentiellement toujours la même classe."
                )

        st.markdown("##### Courbes de loss par fold")
        fig = go.Figure()
        for fold in cv.folds:
            epochs_axis = list(range(len(fold.train_losses)))
            fig.add_trace(go.Scatter(x=epochs_axis, y=fold.train_losses, mode="lines",
                                     name=f"Fold {fold.fold} train", line=dict(dash="solid")))
            fig.add_trace(go.Scatter(x=epochs_axis, y=fold.val_losses, mode="lines",
                                     name=f"Fold {fold.fold} val", line=dict(dash="dot")))
        fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss", height=380,
                          margin=dict(l=40, r=20, t=30, b=40))
        st.plotly_chart(fig, use_container_width=True)

        df = pd.DataFrame([
            {"fold": f.fold, "accuracy": f.accuracy, "auc": f.auc, "f1": f.f1, "best_epoch": f.best_epoch}
            for f in cv.folds
        ])
        st.dataframe(df, hide_index=True, use_container_width=True)


with tab_final:
    st.write(
        "Entraîne un modèle final sur l'ensemble du dataset (avec hold-out interne pour "
        "l'early stopping) et sauvegarde les poids pour la page **Predictions**."
    )
    if st.button("Entraîner et sauvegarder le modèle final", type="primary"):
        lstm_cfg, train_cfg = _configs()
        with st.spinner("Entraînement…"):
            model, tr, va = fit_final_model(x, y, lstm_cfg=lstm_cfg, train_cfg=train_cfg)
            save_model(model, MODEL_PATH)
            if from_db and choix.variant is not None:
                # The sidecar is what lets Predictions rebuild these exact inputs,
                # and it is written by the same function the training CLI uses.
                from src.models.train_all import contract_for
                from src.models.variants import variant_config

                contract = contract_for(
                    variant_config(choix.variant), dataset, x, bool(zscore)
                )
                sidecar_path(MODEL_PATH).write_text(
                    json.dumps(contract.to_dict(), indent=2), encoding="utf-8"
                )
        st.success(f"Modèle sauvegardé : `{MODEL_PATH}`")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(tr))), y=tr, mode="lines", name="train"))
        fig.add_trace(go.Scatter(x=list(range(len(va))), y=va, mode="lines", name="val"))
        fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss", height=350,
                          margin=dict(l=40, r=20, t=30, b=40))
        st.plotly_chart(fig, use_container_width=True)
