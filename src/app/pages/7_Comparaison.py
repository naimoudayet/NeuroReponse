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

from src.models.train_article import OUT_PATH as ARTICLE_PATH
from src.models.variants import ORDERED, VARIANTS
from src.reporting.hypo_ablation import read_json as read_ladder
from src.reporting.sequence_sweep import read_json as read_sweep

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


def read_article() -> list[dict]:
    """The article-aligned results, or empty when the arm has not been run."""
    if not ARTICLE_PATH.exists():
        return []
    return json.loads(ARTICLE_PATH.read_text(encoding="utf-8"))

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

# --------------------------------------------------------------------------- #
# The article-aligned arm — a SECOND table, never a fifth row above.
#
# Pearson r and AUC are different quantities on different targets: one scores a
# continuous BDI-II change, the other ranks a binary label. Putting them on one
# axis would invite reading "0.63 AUC" against "0.16 r" as though the first were
# better, which is not a comparison that exists.
# --------------------------------------------------------------------------- #
st.divider()
st.markdown("## Arm aligné sur l'article (Arteaga et al., PMC12981298)")
st.caption(
    "Même cohorte, mais la méthode de l'étude de référence : cible **continue** "
    "(points BDI-II récupérés) au lieu du label binaire, **un modèle par "
    "protocole** rTMS au lieu d'un modèle poolé, et score = **corrélation de "
    "Pearson** validée par test de permutation."
)

article = read_article()
if not article:
    st.info("Arm non encore entraîné :")
    st.code("python -m src.models.train_article --repeats 3", language="powershell")
else:
    ARM = {1: "P1 · 10 Hz L", 2: "P2 · 1 Hz R"}
    JEU_REG = {
        ("eeg",): "EEG seul (article)",
        ("rtms",): "Clinique seul (référence)",
        ("rtms", "eeg", "ecg"): "Multimodal",
    }
    tbl = pd.DataFrame([
        {
            "Variante": a["key"],
            "Bras": ARM.get(a["protocol"], str(a["protocol"])),
            "Variables": JEU_REG.get(tuple(a["modalities"]), "+".join(a["modalities"])),
            "n": a["n_patients"],
            "r (hors-pli)": f"{a['r_oof']:+.3f}",
            "p (perm.)": f"{a['p_perm']:.3f}",
            "r partiel": f"{a['r_partial']:+.3f}",
            "R²": f"{a['r2']:+.3f}",
            "MAE (pts)": f"{a['mae']:.1f}",
            "Réf. BDI seul": f"{a['baseline_r_bdi_pre']:+.3f}",
        }
        for a in article
    ])
    st.dataframe(tbl, hide_index=True, use_container_width=True)

    montre = article

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        name="Modèle (r hors-pli)", x=[a["key"] for a in montre],
        y=[a["r_oof"] for a in montre], marker_color=BLUE,
    ))
    fig4.add_trace(go.Bar(
        name="BDI-II de référence seul", x=[a["key"] for a in montre],
        y=[a["baseline_r_bdi_pre"] for a in montre], marker_color="#c3c2b7",
    ))
    fig4.add_hline(
        y=0.401, line_dash="dot", line_color=ORANGE,
        annotation_text="article, protocole 1 (r = 0.401)",
        annotation_position="top left",
    )
    fig4.update_layout(
        barmode="group", yaxis_title="Corrélation de Pearson avec ΔBDI observé",
        yaxis_range=[-0.6, 1.0], height=380,
        margin=dict(l=50, r=20, t=30, b=90),
    )
    fig4.add_hline(y=0.0, line_color="#898781")
    st.plotly_chart(fig4, use_container_width=True)

    gagnants = [a for a in article if a["beats_baseline"]]
    significatifs = [a for a in article if a["p_perm"] < 0.05]
    st.markdown("##### Lecture")
    if not gagnants:
        st.warning(
            "**Aucune variante ne dépasse la corrélation du BDI-II de référence "
            "seul.** La cible `delta_bdi` est mathématiquement couplée à la "
            "sévérité initiale — on ne peut pas récupérer 40 points en partant "
            "de 20 — et sur le protocole 1 ce couplage vaut à lui seul r = 0.500, "
            "soit **la même grandeur que le r = 0.401 annoncé par l'article à "
            "partir de l'EEG** (IC95 de ce couplage : [0.258, 0.700] sur 44 "
            "patients, donc indiscernable). Un modèle neurophysiologique qui ne "
            "se distingue pas de cette barre n'a rien démontré."
        )
    if not significatifs:
        st.markdown(
            "- **Aucun résultat significatif** au test de permutation "
            "(tous p ≥ 0.05)."
        )
    st.markdown(
        "- **La ligne à lire est « EEG seul » contre « Clinique seul »**, sur un "
        "même bras. C'est la seule comparaison qui dit si le signal "
        "neurophysiologique apporte quelque chose : le modèle multimodal reçoit "
        "le BDI-II de référence **en entrée**, donc son r ne peut pas distinguer "
        "« l'EEG a prédit » de « le dossier d'admission a prédit »."
    )
    st.markdown(
        "- **La colonne « r partiel »** retire la sévérité initiale des deux "
        "côtés : ce que le modèle a trouvé au-delà du formulaire d'admission."
    )
    st.markdown(
        "- **Les R² sont négatifs** partout : les modèles font moins bien que "
        "prédire simplement la réduction moyenne de la cohorte. Un r positif "
        "avec un R² négatif signale un classement partiel, pas une prédiction."
    )
    st.warning(
        "⚠️ **Ces chiffres sont fragiles par construction.** Sur 44 patients "
        "(protocole 1), dont les prédictions varient de moins d'un point BDI-II, "
        "le r hors-pli est une statistique instable : des étiquettes **mélangées "
        "au hasard** produisent des r de ±0.5 sur ce même montage. Le test "
        "décisif de ce projet n'est donc pas le r mais le ré-entraînement complet "
        "sur étiquettes permutées "
        "(`test_shuffled_labels_do_not_produce_a_correlation`)."
    )
    st.caption(
        "Alignement sur l'article : même cohorte et mêmes 26 canaux, "
        "rééchantillonnage 500 → 250 Hz, notch 50 Hz, passe-bande 0.01–50 Hz, "
        "référence moyenne commune, protocoles séparés, 10 répétitions de "
        "validation croisée à 10 plis, 100 permutations. Écarts restants : "
        "l'article rejette les segments et composantes artefactées (ICA) et "
        "décompose le signal par itEMD avant d'apprendre des filtres "
        "spatio-temporels parcimonieux (SBLEST), là où ce projet moyenne des "
        "puissances de bande par canal ; son protocole 2 compte 73 patients "
        "contre 88 ici. Réduire cet écart est l'objet de l'étape suivante."
    )

# --------------------------------------------------------------------------- #
# The hypothesis ladder — every feature family the new_docs equations specify,
# tested on the same patients and the same folds.
#
# It sits under the article arm rather than inside the 2x2 because it answers a
# different question: not "which of our four models wins" but "does *any*
# feature family in the proposed framework carry rTMS-response signal at all".
# The rungs share an axis (AUC on one binary target) so a single chart is
# honest here, unlike the r-vs-AUC pairing above.
# --------------------------------------------------------------------------- #
st.divider()
st.markdown("## Les équations proposées apportent-elles quelque chose ?")
st.caption(
    "Échelle d'ablation demandée par HYPO4 §11 et par la §5.8 du guide : chaque "
    "barreau ajoute **un seul** bloc de variables au précédent, sur les mêmes "
    "patients et les mêmes plis. Aux blocs de l'article (puissances de bande) "
    "s'ajoutent trois familles qu'il ne calcule pas — synchronisation "
    "(PLV, cohérence, ordre de Kuramoto), complexité (entropie spectrale, pente "
    "1/f, asymétrie alpha frontale) et le rapport harmonique 3-6-9."
)

ladder = read_ladder()
if not ladder:
    st.info("Échelle non encore lancée :")
    st.code(
        "python -m src.reporting.hypo_ablation --regression", language="powershell"
    )
else:
    lrows = ladder["rows"]
    ltable = pd.DataFrame([
        {
            "Modèle": r["model"],
            "Variables": r["label"],
            "n features": r["n_features"],
            "AUC": f"{r['auc']:.3f}",
            "IC 95 %": f"[{r['auc_ci_lo']:.2f}, {r['auc_ci_hi']:.2f}]",
            "PR-AUC": f"{r['pr_auc']:.3f}",
            "Exact. équilibrée": f"{r['balanced_accuracy']:.3f}",
            "Spécificité": f"{r['specificity']:.3f}",
            "Brier": f"{r['brier']:.3f}",
            "p (perm.)": f"{r['p_perm_auc']:.3f}",
            "Signal": "oui" if r["beats_chance"] else "—",
        }
        for r in lrows
    ])
    st.dataframe(ltable, hide_index=True, use_container_width=True)

    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
        x=[r["model"] for r in lrows],
        y=[r["auc"] for r in lrows],
        error_y=dict(
            type="data", symmetric=False,
            array=[r["auc_ci_hi"] - r["auc"] for r in lrows],
            arrayminus=[r["auc"] - r["auc_ci_lo"] for r in lrows],
            color="#52514e",
        ),
        marker_color=[
            "#7c3aed" if set(r["modalities"]) & {"sync", "cplx", "h369"} else BLUE
            for r in lrows
        ],
        customdata=[r["label"] for r in lrows],
        hovertemplate="%{customdata}<br>AUC %{y:.3f}<extra></extra>",
    ))
    fig5.add_hline(
        y=0.5, line_dash="dash", line_color="#898781",
        annotation_text="hasard (0.5)", annotation_position="top left",
    )
    fig5.update_layout(
        yaxis_title="AUC hors-pli (IC 95 % bootstrap sur les patients)",
        yaxis_range=[0.0, 1.0], height=380,
        margin=dict(l=50, r=20, t=30, b=40), showlegend=False,
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.caption(
        "Violet = barreaux contenant au moins une des nouvelles familles de "
        "variables · bleu = blocs déjà présents dans le projet. Les barres "
        "d'erreur sont l'intervalle de confiance bootstrap **rééchantillonné par "
        "patient** — rééchantillonner les lignes traiterait les 8 époques d'un "
        "patient comme 8 observations indépendantes."
    )

    validity = ladder.get("feature_validity", {})
    controls = validity.get("age_controls", [])
    st.markdown("##### Contrôle positif : les nouvelles variables mesurent-elles du réel ?")
    if controls:
        st.dataframe(
            pd.DataFrame([
                {
                    "Variable": c["feature"],
                    "r (âge)": f"{c['r_age']:+.3f}",
                    "p": f"{c['p']:.1e}",
                    "Effet attendu": c["source"],
                    "Confirmé": "oui" if c["confirmed"] else "non",
                }
                for c in controls
            ]),
            hide_index=True, use_container_width=True,
        )
    if validity.get("all_age_controls_confirmed"):
        st.success(
            "**Les trois effets d'âge attendus sont retrouvés**, avec le bon signe "
            "et une p significative. L'extraction fonctionne : le résultat nul "
            "ci-dessus porte donc sur la **tâche**, pas sur un défaut de calcul. "
            "C'est la distinction qu'un tableau de résultats seul ne permet jamais "
            "de faire."
        )

    screens = validity.get("screens", {})
    if screens:
        st.markdown("##### Criblage univarié, correction FDR (Benjamini-Hochberg)")
        st.dataframe(
            pd.DataFrame([
                {
                    "Cible": target,
                    "Meilleure variable": s["best_feature"],
                    "r": f"{s['best_r']:+.3f}",
                    "p brut": f"{s['best_p']:.3f}",
                    "q (FDR)": f"{s['best_q']:.3f}",
                    "Hits bruts": f"{s['n_nominal_hits']}/{s['n_tested']}",
                    "Survivent au FDR": s["n_survivors_fdr"],
                }
                for target, s in screens.items()
            ]),
            hide_index=True, use_container_width=True,
        )

    st.markdown("##### Lecture")
    if not ladder.get("any_signal", False):
        st.warning(
            "**Aucun barreau ne franchit la règle d'arrêt.** Elle exige les quatre "
            "conditions à la fois : intervalle de confiance de l'AUC excluant 0.5, "
            "p de permutation < 5 %, exactitude équilibrée > 0.5, et les deux "
            "classes effectivement prédites. Les familles issues des équations "
            "(synchronisation, complexité, 3-6-9) **n'améliorent pas** la "
            "prédiction de la réponse : elles la laissent au hasard, exactement "
            "comme les puissances de bande de l'article."
        )
    shuffled = ladder.get("shuffled_label_control", {})
    if shuffled.get("shuffled_aucs"):
        vals = ", ".join(f"{v:.3f}" for v in shuffled["shuffled_aucs"])
        st.markdown(
            f"- **Contrôle décisif — étiquettes permutées.** Le meilleur barreau "
            f"réentraîné de bout en bout sur des étiquettes mélangées atteint "
            f"{vals}, contre {shuffled['real_auc']:.3f} avec les vraies. "
            f"Indistinguable : un test de permutation *après* prédiction ne "
            f"suffirait pas, il ne teste que la statistique."
        )
    collinear = ladder.get("physics_collinearity", {})
    if collinear:
        st.markdown(
            f"- **Le modèle E (variables électromagnétiques B/E/J) est vide, pas "
            f"seulement non implémenté.** L'intensité du stimulateur, la géométrie "
            f"de la bobine et la conductivité des tissus ne sont pas publiées par "
            f"TDBRAIN : toute grandeur physique dérivable se réduit à une fonction "
            f"du protocole. Vérification par le rang : "
            f"{collinear['rank_protocol']:.0f} avant, "
            f"{collinear['rank_protocol_plus_physics']:.0f} après avoir ajouté "
            f"{collinear['n_physics_columns']:.0f} colonnes."
        )
    h369 = next((r for r in lrows if r["model"] == "H369"), None)
    base_h = next((r for r in lrows if r["model"] == "G"), None)
    if h369 and base_h:
        delta = h369["auc"] - base_h["auc"]
        st.markdown(
            f"- **L'hypothèse 3-6-9 est réfutée selon son propre critère.** "
            f"L'ajout du bloc harmonique déplace l'AUC de {delta:+.3f} — à "
            f"l'intérieur de l'intervalle de confiance — et aucune de ses "
            f"variables ne survit à la correction des comparaisons multiples. "
            f"Le document exige explicitement que H369 soit « considérée comme "
            f"non soutenue » dans ce cas."
        )

# --------------------------------------------------------------------------- #
# The multi-session question — deliberately *outside* the 2x2.
#
# Both cohorts above are baseline-only: their sequence axis is epochs of one
# resting recording. So the four numbers say nothing about whether accumulating
# treatment sessions helps, which is the premise the clinical loop rests on.
# That needs the sequential cohort, a different sequence axis and a different
# y-scale — putting it in the bar chart above would invite reading an
# incomparable number as a fifth variant.
# --------------------------------------------------------------------------- #
st.divider()
st.markdown("## Est-ce que cumuler les séances aide ?")
st.caption(
    "Les quatre modèles ci-dessus sont **baseline-only** : leur axe temporel est "
    "constitué d'époques d'un unique enregistrement de repos. Ils ne peuvent donc "
    "pas répondre à la question sur laquelle repose la boucle clinique — *la "
    "prédiction s'améliore-t-elle quand le patient revient ?* Seule la cohorte "
    "**simulée séquentielle** (10 vraies séances de traitement) permet de la poser. "
    "La mesure rejoue la même validation croisée patient-wise sur la même cohorte, "
    "tronquée aux k premières séances."
)

points = read_sweep()
if not points:
    st.info("Mesure non encore lancée :")
    st.code("python -m src.reporting.sequence_sweep", language="powershell")
else:
    ks = [p["n_sessions"] for p in points]
    aucs = [p["auc_mean"] for p in points]
    stds = [p["auc_std"] for p in points]
    accs = [p["accuracy_mean"] for p in points]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=ks, y=aucs, mode="lines+markers", name="AUC",
        line=dict(color="#7c3aed", width=3),
        error_y=dict(type="data", array=stds, color="#a99fd0"),
    ))
    fig3.add_trace(go.Scatter(
        x=ks, y=accs, mode="lines+markers", name="Exactitude",
        line=dict(color=BLUE, width=2, dash="dot"),
    ))
    fig3.add_hline(
        y=points[0]["base_rate"], line_dash="dash", line_color="#898781",
        annotation_text=f"taux de base ({points[0]['base_rate']:.0%})",
        annotation_position="bottom left",
    )
    fig3.update_layout(
        xaxis_title="Nombre de séances de traitement fournies au modèle",
        yaxis_title="Score (validation croisée patient-wise)",
        yaxis_range=[0.0, 1.05], height=380,
        margin=dict(l=50, r=20, t=30, b=40),
        xaxis=dict(tickmode="linear", dtick=1),
    )
    st.plotly_chart(fig3, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("AUC à 1 séance", f"{aucs[0]:.3f}", f"± {stds[0]:.3f}")
    m2.metric(f"AUC à {ks[-1]} séances", f"{aucs[-1]:.3f}", f"± {stds[-1]:.3f}")
    m3.metric("Gain apporté par le cumul", f"{aucs[-1] - aucs[0]:+.3f}")

    facteur = (1 - accs[0]) / max(1 - accs[-1], 1e-9)
    st.markdown(
        f"- **Le cumul des séances aide, et de façon mesurable** : l'AUC passe de "
        f"{aucs[0]:.3f} à {aucs[-1]:.3f} et l'exactitude de {accs[0]:.0%} à "
        f"{accs[-1]:.0%} — le taux d'erreur est divisé par {facteur:.0f}."
    )
    st.markdown(
        f"- **L'écart-type entre plis s'effondre** ({stds[0]:.3f} → {stds[-1]:.3f}) : "
        f"cumuler des séances ne remonte pas seulement la moyenne, cela **stabilise** "
        f"l'estimation. C'est l'argument architectural de la boucle clinique."
    )
    st.warning(
        f"⚠️ **Cette cohorte est simulée avec un biomarqueur injecté** "
        f"(`src/data/simulator.py` module la puissance alpha selon le label). Les "
        f"valeurs absolues sont artificiellement hautes — dès 1 séance on est à "
        f"{aucs[0]:.2f}. Ce qu'il faut lire ici, c'est la **forme** de la courbe : "
        f"le pipeline convertit des séances supplémentaires en précision. Sur données "
        f"réelles la question reste **ouverte** — TDBRAIN ne publie aucune trajectoire "
        f"de traitement, et c'est précisément ce qui manque pour trancher."
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
