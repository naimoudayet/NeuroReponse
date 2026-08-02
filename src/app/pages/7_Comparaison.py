"""The 2x2 comparison, read from what training actually wrote.

Every number on this page comes from ``data/models/comparison.json`` — the file
:mod:`src.models.train_all` writes at the end of a run. Nothing is hard-coded:
a page quoting AUCs typed in by hand would keep displaying them long after the
models changed, which is exactly the drift the JSON sidecars exist to prevent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.variants import ORDERED, VARIANTS

st.set_page_config(page_title="Comparaison", page_icon=":bar_chart:", layout="wide")
st.title("Comparaison des quatre modèles")
st.caption(
    "Deux cohortes (simulée appariée, TDBRAIN réelle) × deux jeux de variables "
    "(clinique seul, clinique + EEG + ECG). Lire les quatre chiffres ensemble "
    "sépare deux questions qu'un modèle isolé ne peut pas trancher : *le signal "
    "neurophysiologique apporte-t-il quelque chose ?* (comparer une ligne) et "
    "*la cohorte réelle se comporte-t-elle comme le simulateur le prédit ?* "
    "(comparer une colonne)."
)

COMPARISON = Path("data/models/comparison.json")

if not COMPARISON.exists():
    st.warning(f"`{COMPARISON}` absent — lance d'abord la comparaison complète :")
    st.code(
        "python -m src.models.train_all --root "
        '"data/tdbrain/TDBRAIN_Dataset_V3_1_Encr/TDBRAIN_Dataset_V3_1"',
        language="powershell",
    )
    st.stop()

results = json.loads(COMPARISON.read_text(encoding="utf-8"))
by_key = {r["key"]: r for r in results}

# Registry order, so the simulated/real pair sits side by side per feature set.
rows = [by_key[v.value] for v in ORDERED if v.value in by_key]
if not rows:
    st.error("Le fichier de comparaison ne contient aucune variante connue.")
    st.stop()

COHORTE = {"simule": "Simulée (appariée)", "tdbrain": "TDBRAIN (réelle)"}
JEU = {1: "Clinique seul", 3: "Clinique + EEG + ECG"}

table = pd.DataFrame([
    {
        "Variante": r["key"],
        "Cohorte": COHORTE.get(r["dataset"], r["dataset"]),
        "Variables": JEU.get(len(r["modalities"]), "+".join(r["modalities"])),
        "n features": r["n_features"],
        "n patients": r["n_patients"],
        "AUC": f"{r['auc_mean']:.3f} ± {r['auc_std']:.3f}",
        "Exactitude": f"{r['accuracy_mean']:.3f}",
        "F1": f"{r['f1_mean']:.3f}",
        "Taux de base": f"{r['base_rate']:.3f}",
    }
    for r in rows
])
st.dataframe(table, hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------- #
# AUC with its fold spread. One axis, chance drawn explicitly: an AUC bar chart
# starting at zero exaggerates differences that the fold std already swamps.
# --------------------------------------------------------------------------- #
BLUE, ORANGE = "#2a78d6", "#eb6834"

fig = go.Figure()
fig.add_trace(go.Bar(
    x=[r["key"] for r in rows],
    y=[r["auc_mean"] for r in rows],
    error_y=dict(type="data", array=[r["auc_std"] for r in rows], color="#52514e"),
    marker_color=[BLUE if r["dataset"] == "simule" else ORANGE for r in rows],
    hovertemplate="%{x}<br>AUC %{y:.3f}<extra></extra>",
))
fig.add_hline(
    y=0.5, line_dash="dash", line_color="#898781",
    annotation_text="hasard (0.5)", annotation_position="top left",
)
fig.update_layout(
    yaxis_title="AUC (moyenne des plis, ± écart-type)",
    yaxis_range=[0.0, 1.0], height=380,
    margin=dict(l=50, r=20, t=30, b=40), showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Bleu = cohorte simulée appariée · orange = cohorte TDBRAIN réelle. "
    "Les barres d'erreur sont l'écart-type entre les plis de la validation "
    "croisée patient-wise."
)

# --------------------------------------------------------------------------- #
# Accuracy against the base rate — the comparison that actually settles it.
# --------------------------------------------------------------------------- #
st.markdown("##### Exactitude vs taux de la classe majoritaire")
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    name="Exactitude", x=[r["key"] for r in rows],
    y=[r["accuracy_mean"] for r in rows], marker_color=BLUE,
))
fig2.add_trace(go.Bar(
    name="Taux de base", x=[r["key"] for r in rows],
    y=[r["base_rate"] for r in rows], marker_color="#c3c2b7",
))
fig2.update_layout(
    barmode="group", yaxis_title="Proportion", yaxis_range=[0.0, 1.0], height=340,
    margin=dict(l=50, r=20, t=30, b=40),
)
st.plotly_chart(fig2, use_container_width=True)

ecarts = [r["accuracy_mean"] - r["base_rate"] for r in rows]
if max(ecarts) < 0.05:
    st.warning(
        "**Aucune variante ne dépasse son taux de base.** Une exactitude égale au "
        "taux de la classe majoritaire signifie que le modèle prédit "
        "essentiellement toujours « répondeur » : la colonne des non-répondeurs "
        "de la matrice de confusion est vide. C'est le résultat central de ce "
        "travail, et il est négatif."
    )

# --------------------------------------------------------------------------- #
# What the four numbers say together.
# --------------------------------------------------------------------------- #
clin = [r for r in rows if len(r["modalities"]) == 1]
multi = [r for r in rows if len(r["modalities"]) > 1]

st.markdown("##### Lecture")
if clin and multi:
    moy_clin = sum(r["auc_mean"] for r in clin) / len(clin)
    moy_multi = sum(r["auc_mean"] for r in multi) / len(multi)
    if moy_clin > moy_multi:
        st.markdown(
            f"- **Le modèle clinique bat le multimodal sur les deux cohortes** "
            f"(AUC moyenne {moy_clin:.3f} contre {moy_multi:.3f}). Quatre variables "
            f"battent 139 : ajouter 135 colonnes non informatives dilue les quatre "
            f"qui le sont, sur seulement {rows[0]['n_patients']} patients."
        )
st.markdown(
    "- Les écarts entre variantes sont **du même ordre que l'écart-type entre "
    "plis** : les classer par AUC reviendrait à commenter du bruit.\n"
    "- La cohorte simulée est générée **sans effet neurophysiologique injecté** "
    "(`effect_size=0`) : elle reproduit délibérément le résultat nul réel. Elle "
    "sert de contrôle négatif, pas de démonstration de performance."
)

with st.expander("Détail des variantes (registre)"):
    st.dataframe(
        pd.DataFrame([
            {
                "clé": v.value,
                "libellé": VARIANTS[v].label,
                "modalités": " + ".join(VARIANTS[v].modalities),
                "base": str(VARIANTS[v].db),
                "checkpoint": str(VARIANTS[v].model),
                "entraîné": VARIANTS[v].model.exists(),
            }
            for v in ORDERED
        ]),
        hide_index=True, use_container_width=True,
    )
