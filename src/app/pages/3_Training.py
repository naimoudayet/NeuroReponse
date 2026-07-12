from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app.utils import MODEL_PATH, SIM_DIR, has_trained_model
from src.data.loader import load
from src.models.lstm import LSTMConfig
from src.models.train import TrainConfig, cross_validate, fit_final_model, save_model
from src.preprocessing.pipeline import PipelineConfig, preprocess

st.set_page_config(page_title="Training", page_icon=":robot_face:", layout="wide")
st.title("Entraînement du modèle LSTM")

if not (SIM_DIR / "eeg_simulated.npz").exists():
    st.error(
        "Le dataset simulé n'existe pas encore. Lance d'abord l'initialisation depuis l'accueil "
        "(ou: `python -m src.data.simulator`)."
    )
    st.stop()

ds = load(SIM_DIR)
c1, c2, c3 = st.columns(3)
c1.metric("Patients", ds.signals.shape[0])
c2.metric("Sessions / patient", ds.signals.shape[1])
c3.metric("Modèle existant", "oui" if has_trained_model() else "non")

st.markdown("##### Hyperparamètres")
left, right = st.columns(2)
with left:
    epochs = st.slider("Epochs", 5, 60, 30)
    batch_size = st.select_slider("Batch size", options=[4, 8, 16, 32], value=8)
    lr = st.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=3e-3, format_func=lambda v: f"{v:.0e}")
with right:
    n_splits = st.slider("Folds (CV patient-wise)", 3, 10, 5)
    dropout = st.slider("Dropout", 0.0, 0.6, 0.3, step=0.05)
    seed = st.number_input("Seed", 0, 1000, 0)

st.divider()


def _preprocess(ds_):
    return preprocess(ds_.signals, PipelineConfig(fs=ds_.fs, mode="features"))


tab_cv, tab_final = st.tabs(["Validation croisée", "Modèle final"])


with tab_cv:
    if st.button("Lancer la validation croisée patient-wise", type="primary"):
        pre = _preprocess(ds)
        groups = np.arange(pre.x.shape[0])
        lstm_cfg = LSTMConfig(input_size=pre.x.shape[-1], hidden_sizes=(128, 64), dropout=dropout)
        train_cfg = TrainConfig(epochs=epochs, batch_size=batch_size, lr=lr,
                                early_stopping_patience=max(3, epochs // 6), seed=seed)

        with st.spinner("Entraînement en cours…"):
            cv = cross_validate(pre.x, ds.labels.astype(np.float32), groups,
                                lstm_cfg=lstm_cfg, train_cfg=train_cfg, n_splits=n_splits)
        st.session_state["last_cv"] = cv

    cv = st.session_state.get("last_cv")
    if cv:
        summary = cv.summary()
        a, b, c = st.columns(3)
        a.metric("Accuracy", f"{summary['accuracy_mean']:.2%}", f"± {summary['accuracy_std']:.2%}")
        b.metric("AUC", f"{summary['auc_mean']:.2%}", f"± {summary['auc_std']:.2%}")
        c.metric("F1", f"{summary['f1_mean']:.2%}", f"± {summary['f1_std']:.2%}")

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
        "Entraîne un modèle final sur l'ensemble du dataset (avec hold-out interne pour l'early stopping) "
        "et sauvegarde les poids pour la page **Predictions**."
    )
    if st.button("Entraîner et sauvegarder le modèle final", type="primary"):
        pre = _preprocess(ds)
        lstm_cfg = LSTMConfig(input_size=pre.x.shape[-1], hidden_sizes=(128, 64), dropout=dropout)
        train_cfg = TrainConfig(epochs=epochs, batch_size=batch_size, lr=lr,
                                early_stopping_patience=max(3, epochs // 6), seed=seed)
        with st.spinner("Entraînement…"):
            model, tr, va = fit_final_model(pre.x, ds.labels.astype(np.float32),
                                            lstm_cfg=lstm_cfg, train_cfg=train_cfg)
            save_model(model, MODEL_PATH)
        st.success(f"Modèle sauvegardé : `{MODEL_PATH}`")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(tr))), y=tr, mode="lines", name="train"))
        fig.add_trace(go.Scatter(x=list(range(len(va))), y=va, mode="lines", name="val"))
        fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss", height=350,
                          margin=dict(l=40, r=20, t=30, b=40))
        st.plotly_chart(fig, use_container_width=True)
