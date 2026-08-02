"""Generate the four per-model notebooks.

    python -m src.reporting.build_notebooks

The four reports are structurally identical and differ only by variant, so they
are generated from one template rather than maintained as four hand-edited
copies: four copies drift, and a chart that differs between two notebooks invites
a comparison that is no longer like-for-like.

Each notebook is self-contained and executable without the gated TDBRAIN
download — the real-cohort notebooks fall back to the synthetic fixture and say
so loudly at the top of the output.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from ..models.variants import ORDERED, VARIANTS, Dataset, Variant, VariantConfig

NOTEBOOK_DIR = Path("notebooks")

# notebook number per variant, following the app's presentation order
NUMBERS: dict[Variant, str] = {
    Variant.SIM_RTMS: "05",
    Variant.TDBRAIN_RTMS: "06",
    Variant.SIM_MULTI: "07",
    Variant.TDBRAIN_MULTI: "08",
}

SLUGS: dict[Variant, str] = {
    Variant.SIM_RTMS: "model_sim_rtms",
    Variant.TDBRAIN_RTMS: "model_tdbrain_rtms",
    Variant.SIM_MULTI: "model_sim_multi",
    Variant.TDBRAIN_MULTI: "model_tdbrain_multi",
}


# nbformat >= 4.5 wants a stable id on every cell and will make its absence a
# hard error; a simple counter keeps them unique and reproducible across runs.
_CELL_N = itertools.count()


def _md(*lines: str) -> dict:
    return {
        "cell_type": "markdown", "id": f"md{next(_CELL_N):03d}",
        "metadata": {}, "source": list(lines),
    }


def _code(*lines: str) -> dict:
    return {
        "cell_type": "code", "id": f"cd{next(_CELL_N):03d}", "execution_count": None,
        "metadata": {}, "outputs": [], "source": list(lines),
    }


def _header_cells(cfg: VariantConfig) -> list[dict]:
    is_sim = cfg.dataset is Dataset.SIMULE
    is_multi = "eeg" in cfg.modalities
    cohort = "simulée (calibrée sur TDBRAIN)" if is_sim else "réelle TDBRAIN"
    feats = (
        "**clinique seul** : protocole rTMS, âge, sexe, BDI-II de référence (4 variables)"
        if not is_multi else
        "**multimodal** : 4 variables cliniques + 130 puissances de bande EEG "
        "(26 canaux × 5 bandes) + 5 métriques HRV issues de l'ECG (139 variables)"
    )
    return [
        _md(
            f"# {NUMBERS[cfg.key]} — {cfg.label}\n",
            "\n",
            f"Évaluation du modèle **{cfg.key.value}** : cohorte {cohort}, "
            f"jeu de variables {feats}.\n",
            "\n",
            "Ce carnet fait partie d'une série de quatre, qui croise **deux cohortes** "
            "(simulée, réelle) et **deux jeux de variables** (clinique, multimodal). "
            "Lire les quatre ensemble sépare deux questions qu'un modèle seul ne peut "
            "pas distinguer : *le signal neurophysiologique apporte-t-il quelque "
            "chose ?* (comparer les jeux de variables) et *la cohorte simulée "
            "reproduit-elle la réalité ?* (comparer les cohortes).\n",
            "\n",
            "| Carnet | Cohorte | Variables |\n",
            "|---|---|---|\n",
            *[
                f"| {NUMBERS[v]} | "
                f"{'simulée' if VARIANTS[v].dataset is Dataset.SIMULE else 'TDBRAIN'} | "
                f"{'clinique' if 'eeg' not in VARIANTS[v].modalities else 'multimodal'} |\n"
                for v in ORDERED
            ],
            "\n",
            "> **Protocole d'évaluation** — validation croisée **patient-wise** "
            "(`GroupKFold`) : aucun patient n'apparaît à la fois en entraînement et "
            "en validation. Toutes les courbes sont tracées sur les prédictions "
            "**hors échantillon**.\n",
        ),
        _code(
            "import sys, warnings\n",
            "from pathlib import Path\n",
            "\n",
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
            "if str(ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(ROOT))\n",
            "\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "from src.data.modalities import build_features\n",
            "from src.models.lstm import LSTMConfig\n",
            "from src.models.train import TrainConfig, cross_validate\n",
            "from src.models.variants import Variant, variant_config\n",
            "from src.reporting import model_charts as mc\n",
            "\n",
            f"CFG = variant_config(Variant.{cfg.key.name})\n",
            "print(CFG.label, '|', '+'.join(CFG.modalities))\n",
        ),
    ]


def _load_cells(cfg: VariantConfig) -> list[dict]:
    if cfg.dataset is Dataset.SIMULE:
        return [
            _md(
                "## 1. La cohorte\n",
                "\n",
                "Cohorte synthétique **calibrée sur TDBRAIN** : même effectif (132), "
                "même montage (26 canaux), même taux de répondeurs (63 %), et des "
                "distributions cliniques ajustées sur les valeurs réelles.\n",
                "\n",
                "`effect_size` contrôle le signal injecté dans l'EEG et l'ECG. "
                "À **0**, ces blocs ne contiennent aucune information sur la réponse — "
                "ce qui reproduit le résultat réel. C'est un **contrôle positif** : "
                "il permet de vérifier que le pipeline ne fabrique pas de signal.\n",
            ),
            _code(
                "from src.data.simulator_matched import MatchedSimConfig, simulate_matched\n",
                "\n",
                "EFFECT = 0.0   # 0 = reproduit le résultat réel ; >0 = effet injecté\n",
                "ds = simulate_matched(MatchedSimConfig(effect_size=EFFECT, seed=42))\n",
                "print(f'{ds.signals_mc.shape[0]} patients · {ds.signals_mc.shape[1]} époques '\n",
                "      f'· {ds.signals_mc.shape[2]} canaux · fs {ds.fs:g} Hz')\n",
                "print(f'répondeurs : {int(ds.labels.sum())}/{len(ds.labels)} '\n",
                "      f'({ds.labels.mean():.1%})')\n",
            ),
        ]
    return [
        _md(
            "## 1. La cohorte\n",
            "\n",
            "Cohorte **réelle** TDBRAIN : 132 patients dépressifs traités par rTMS "
            "(protocoles 1 et 2), un enregistrement de repos avant traitement par "
            "patient, 26 canaux EEG + dérivation ECG.\n",
            "\n",
            "> Les données sont soumises à un accord d'utilisation et ne sont pas "
            "dans le dépôt. Si elles sont absentes, ce carnet bascule sur le jeu "
            "**synthétique de test** afin de rester exécutable — les chiffres ne "
            "sont alors pas ceux de la cohorte réelle, et un avertissement le "
            "signale.\n",
        ),
        _code(
            "from src.data.tdbrain import (\n",
            "    TDBRAINConfig, load_tdbrain, make_synthetic_tdbrain,\n",
            ")\n",
            "\n",
            "TDBRAIN_ROOT = ROOT / 'data/tdbrain/TDBRAIN_Dataset_V3_1_Encr/TDBRAIN_Dataset_V3_1'\n",
            "REAL = (TDBRAIN_ROOT / 'participants.tsv').exists()\n",
            "\n",
            "with warnings.catch_warnings():\n",
            "    warnings.simplefilter('ignore')\n",
            "    if REAL:\n",
            "        ds = load_tdbrain(TDBRAINConfig(\n",
            "            root=TDBRAIN_ROOT, col_id='TDBRAIN_ID',\n",
            "            col_protocol='rTMS PROTOCOL',\n",
            "        ))\n",
            "    else:\n",
            "        import tempfile\n",
            "        print('!! DONNEES REELLES ABSENTES — repli sur le jeu synthetique !!')\n",
            "        root = Path(tempfile.mkdtemp()) / 'td'\n",
            "        make_synthetic_tdbrain(root, n_patients=40, seed=1,\n",
            "                               with_ecg=True, duration_seconds=20.0)\n",
            "        ds = load_tdbrain(TDBRAINConfig(\n",
            "            root=root, n_epochs=4, epoch_seconds=1.0, target_fs=250.0,\n",
            "        ))\n",
            "\n",
            "print('cohorte reelle' if REAL else 'JEU SYNTHETIQUE (demonstration)')\n",
            "print(f'{ds.signals_mc.shape[0]} patients · {ds.signals_mc.shape[1]} époques '\n",
            "      f'· {ds.signals_mc.shape[2]} canaux · fs {ds.fs:g} Hz')\n",
            "print(f'répondeurs : {int(ds.labels.sum())}/{len(ds.labels)} '\n",
            "      f'({ds.labels.mean():.1%})')\n",
        ),
    ]


def _body_cells(cfg: VariantConfig) -> list[dict]:
    is_multi = "eeg" in cfg.modalities
    cells = [
        _md(
            "### Composition de la cohorte\n",
            "\n",
            "Le **taux de base** — la proportion de la classe majoritaire — est la "
            "référence à battre. Un modèle qui prédit toujours « répondeur » "
            "atteint mécaniquement cette exactitude sans rien avoir appris.\n",
        ),
        _code(
            "md = ds.metadata\n",
            "base_rate = float(max(ds.labels.mean(), 1 - ds.labels.mean()))\n",
            "\n",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), constrained_layout=True)\n",
            "fig.patch.set_facecolor(mc.SURFACE)\n",
            "\n",
            "# Répartition des classes\n",
            "counts = [int((ds.labels == 0).sum()), int((ds.labels == 1).sum())]\n",
            "axes[0].bar(['Non-répondeur', 'Répondeur'], counts,\n",
            "            color=[mc.SERIES[1], mc.SERIES[0]], width=0.6)\n",
            "for i, c in enumerate(counts):\n",
            "    axes[0].annotate(str(c), xy=(i, c), xytext=(0, 4),\n",
            "                     textcoords='offset points', ha='center',\n",
            "                     fontsize=10, fontweight='bold', color=mc.INK)\n",
            "mc.style_axes(axes[0], f'Classes (taux de base {base_rate:.1%})', '', 'Patients')\n",
            "\n",
            "# Âge par classe — la variable la plus discriminante de la cohorte réelle\n",
            "for lbl, name, colour in ((1, 'Répondeur', mc.SERIES[0]),\n",
            "                          (0, 'Non-répondeur', mc.SERIES[1])):\n",
            "    axes[1].hist(md.loc[ds.labels == lbl, 'age'], bins=12, alpha=0.65,\n",
            "                 color=colour, label=name)\n",
            "leg = axes[1].legend(frameon=False, fontsize=9)\n",
            "for t in leg.get_texts():\n",
            "    t.set_color(mc.INK_2)\n",
            "mc.style_axes(axes[1], 'Âge par classe', 'Âge (années)', 'Patients')\n",
            "\n",
            "# BDI-II de référence par classe — ne sépare pas, sur données réelles\n",
            "for lbl, name, colour in ((1, 'Répondeur', mc.SERIES[0]),\n",
            "                          (0, 'Non-répondeur', mc.SERIES[1])):\n",
            "    axes[2].hist(md.loc[ds.labels == lbl, 'bdi_pre'], bins=12, alpha=0.65,\n",
            "                 color=colour, label=name)\n",
            "leg = axes[2].legend(frameon=False, fontsize=9)\n",
            "for t in leg.get_texts():\n",
            "    t.set_color(mc.INK_2)\n",
            "mc.style_axes(axes[2], 'BDI-II de référence par classe', 'BDI-II', 'Patients')\n",
            "plt.show()\n",
            "\n",
            "print(f\"âge      — répondeurs {md.loc[ds.labels==1,'age'].mean():.1f} \"\n",
            "      f\"vs non-répondeurs {md.loc[ds.labels==0,'age'].mean():.1f}\")\n",
            "print(f\"BDI_pre  — répondeurs {md.loc[ds.labels==1,'bdi_pre'].mean():.1f} \"\n",
            "      f\"vs non-répondeurs {md.loc[ds.labels==0,'bdi_pre'].mean():.1f}\")\n",
        ),
        _md(
            "## 2. Construction des variables\n",
            "\n",
            "Les blocs sont assemblés par `src.data.modalities.build_features`, "
            "utilisée à l'identique pour les quatre modèles. L'ordre des blocs est "
            "**canonique** (`rtms, eeg, ecg`) quel que soit l'ordre demandé : un "
            "vecteur permuté serait silencieux et fatal.\n",
            "\n",
            "Les blocs **clinique** et **HRV** sont constants d'une époque à l'autre "
            "(ce sont des propriétés du patient) : ils ne sont **jamais** normalisés "
            "par z-score intra-patient, qui les réduirait à zéro. Seul le bloc EEG "
            "l'est.\n",
        ),
        _code(
            "x, y, groups, names = build_features(\n",
            "    ds, modalities=CFG.modalities, per_patient_zscore=True,\n",
            ")\n",
            "print(f'x = {x.shape}  (patients, époques, variables)')\n",
            "print(f'{len(names)} variables — {names[:4]} … {names[-3:]}')\n",
        ),
        _md(
            "## 3. Validation croisée\n",
            "\n",
            "`GroupKFold` sur l'identifiant patient. Les probabilités hors "
            "échantillon de chaque pli sont conservées : chaque patient est ainsi "
            "noté exactement une fois, par un modèle qui ne l'a jamais vu.\n",
        ),
        _code(
            "cv = cross_validate(\n",
            "    x, y.astype(np.float32), groups,\n",
            "    lstm_cfg=LSTMConfig(input_size=x.shape[-1]),\n",
            "    train_cfg=TrainConfig(epochs=30),\n",
            "    n_splits=5,\n",
            ")\n",
            "summary = cv.summary()\n",
            "for k in ('auc_mean', 'auc_std', 'accuracy_mean', 'f1_mean'):\n",
            "    print(f'{k:<15} {summary[k]:.4f}')\n",
            "print(f'{\"taux de base\":<15} {base_rate:.4f}')\n",
        ),
        _md(
            "## 4. Rapport d'évaluation\n",
            "\n",
            "Six panneaux. Deux points de lecture :\n",
            "\n",
            "- **Deux AUC apparaissent, et elles diffèrent.** Le panneau de verdict "
            "montre la *moyenne des AUC par pli* ; la courbe ROC montre l'*AUC "
            "groupée* sur toutes les prédictions hors échantillon. Ce sont deux "
            "quantités légitimes et différentes.\n",
            "- **L'exactitude se lit contre le taux de base.** Une exactitude égale au "
            "taux de base signifie que le modèle prédit toujours la même classe — la "
            "matrice de confusion le rend visible (une colonne vide).\n",
        ),
        _code(
            "fig = mc.model_report(\n",
            "    CFG.label, cv, y, x, groups, names, base_rate=base_rate,\n",
            ")\n",
            "plt.show()\n",
        ),
        _md(
            "### Courbes d'apprentissage\n",
            "\n",
            "Perte d'entraînement et de validation pour chaque pli. Un écart qui se "
            "creuse indique un surapprentissage ; deux courbes plates indiquent que "
            "le modèle n'a rien trouvé à apprendre.\n",
        ),
        _code(
            "fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)\n",
            "fig.patch.set_facecolor(mc.SURFACE)\n",
            "mc.plot_learning_curves(ax, cv.folds)\n",
            "plt.show()\n",
        ),
    ]

    if cfg.dataset is Dataset.SIMULE and is_multi:
        cells += [
            _md(
                "## 5. Contrôle positif\n",
                "\n",
                "La question que ce carnet doit trancher : le pipeline est-il "
                "*incapable* de détecter un signal, ou n'y en a-t-il *pas* ?\n",
                "\n",
                "On régénère la même cohorte avec un effet injecté dans l'EEG et "
                "l'ECG. Si l'AUC monte, le pipeline fonctionne — et le résultat nul "
                "obtenu à `effect_size=0` décrit bien les données, pas un défaut de "
                "traitement.\n",
            ),
            _code(
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.model_selection import StratifiedKFold, cross_val_score\n",
                "from sklearn.pipeline import make_pipeline\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "\n",
                "def quick_auc(dataset):\n",
                "    xx, yy, _, _ = build_features(\n",
                "        dataset, modalities=('eeg', 'ecg'), per_patient_zscore=False,\n",
                "    )\n",
                "    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))\n",
                "    scores = cross_val_score(\n",
                "        model, xx.mean(axis=1), yy, scoring='roc_auc',\n",
                "        cv=StratifiedKFold(5, shuffle=True, random_state=0),\n",
                "    )\n",
                "    return scores.mean()\n",
                "\n",
                "effects = [0.0, 0.15, 0.30, 0.50]\n",
                "aucs = []\n",
                "for e in effects:\n",
                "    d = simulate_matched(MatchedSimConfig(effect_size=e, seed=7))\n",
                "    aucs.append(quick_auc(d))\n",
                "    print(f'effect_size={e:.2f} -> AUC {aucs[-1]:.3f}')\n",
            ),
            _code(
                "fig, ax = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)\n",
                "fig.patch.set_facecolor(mc.SURFACE)\n",
                "ax.axhline(0.5, color=mc.MUTED, linewidth=1.2, linestyle='--')\n",
                "# Right-aligned: the left edge is where the effect_size=0 point and its\n",
                "# value label sit, and the three would collide.\n",
                "ax.annotate('hasard (0.5)', xy=(0.98, 0.515), xycoords=('axes fraction', 'data'),\n",
                "            ha='right', color=mc.MUTED, fontsize=8)\n",
                "ax.plot(effects, aucs, color=mc.SERIES[0], linewidth=2.0,\n",
                "        marker='o', markersize=9, markeredgecolor=mc.SURFACE, markeredgewidth=2)\n",
                "for e, a in zip(effects, aucs):\n",
                "    ax.annotate(f'{a:.2f}', xy=(e, a), xytext=(0, 10),\n",
                "                textcoords='offset points', ha='center',\n",
                "                fontsize=9, color=mc.INK, fontweight='bold')\n",
                "ax.annotate('réel ≈ ici', xy=(0.0, aucs[0]), xytext=(6, -18),\n",
                "            textcoords='offset points', fontsize=9, color=mc.INK_2)\n",
                "ax.set_ylim(0.3, 1.0)\n",
                "mc.style_axes(ax, \"Contrôle positif : le pipeline détecte-t-il un effet injecté ?\",\n",
                "              \"Taille d'effet injectée dans l'EEG/ECG\", 'AUC')\n",
                "plt.show()\n",
            ),
        ]

    cells.append(
        _md(
            "## Conclusion\n",
            "\n",
            "À compléter à la lecture des sorties ci-dessus. Les trois questions à "
            "trancher :\n",
            "\n",
            "1. L'AUC dépasse-t-elle 0,5 **en tenant compte de la dispersion entre "
            "plis** ? Une moyenne au-dessus du hasard dont l'écart-type recouvre "
            "0,5 n'établit rien sur cet effectif.\n",
            "2. L'exactitude dépasse-t-elle le **taux de base** ? Sinon, le modèle "
            "prédit toujours la classe majoritaire.\n",
            "3. Le résultat est-il **cohérent avec le carnet apparié** (même jeu de "
            "variables, autre cohorte) ?\n",
            "\n",
            "> Un résultat négatif obtenu sous un protocole propre est un résultat "
            "publiable. Il ne doit pas être masqué par un choix de seuil ou une "
            "métrique flatteuse.\n",
        )
    )
    return cells


def build_notebook(cfg: VariantConfig) -> Path:
    nb = {
        "cells": _header_cells(cfg) + _load_cells(cfg) + _body_cells(cfg),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOK_DIR / f"{NUMBERS[cfg.key]}_{SLUGS[cfg.key]}.ipynb"
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def build_all() -> list[Path]:
    return [build_notebook(VARIANTS[v]) for v in ORDERED]


if __name__ == "__main__":
    for p in build_all():
        print(f"wrote {p}")
