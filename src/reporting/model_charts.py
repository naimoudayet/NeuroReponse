"""Evaluation charts for the four models, shared by the notebooks.

Written once and imported by every notebook so the four reports are visually and
methodologically identical — a chart drawn slightly differently per notebook
would invite comparisons that are not like-for-like.

**Palette.** The categorical slots, ink tokens and status colours below are the
validated defaults (blue / orange / aqua, checked for colour-vision separation
against the light chart surface). Two rules the palette imposes and this module
honours: aqua sits under 3:1 contrast on the surface, so anything drawn in it
carries a visible direct label; and status colours are always paired with a word,
never left to carry meaning alone.

**One axis per chart, always.** Where two quantities of different scale matter,
they get two panels rather than a second y-axis.

Every curve is drawn from **out-of-fold** predictions: with GroupKFold each
patient is scored exactly once, by a model that never saw them in training. An
ROC drawn on training predictions would look far better and mean nothing.
"""
from __future__ import annotations

import textwrap

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# --- validated palette ------------------------------------------------------ #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")     # blue, orange, aqua — fixed order
LOW_CONTRAST = {"#1baf7a"}                      # needs a direct label

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# Sequential blue ramp, light -> dark, for magnitude (confusion matrix).
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

CHANCE = 0.5


def style_axes(ax: Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Recessive chrome: hairline grid, muted labels, no top/right spines."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if title:
        ax.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


# --------------------------------------------------------------------------- #
# Discrimination
# --------------------------------------------------------------------------- #


def plot_roc(ax: Axes, y_true: np.ndarray, y_proba: np.ndarray, label: str = "Modèle") -> float:
    """ROC from out-of-fold predictions, against the chance diagonal."""
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = float(roc_auc_score(y_true, y_proba))

    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle="--", zorder=1)
    ax.annotate("hasard", xy=(0.62, 0.58), color=MUTED, fontsize=8, rotation=38)
    ax.plot(fpr, tpr, color=SERIES[0], linewidth=2.0, zorder=3)
    ax.fill_between(fpr, tpr, alpha=0.10, color=SERIES[0], zorder=2)

    # Direct label rather than a legend box: one series, so the title names it.
    # "groupée" matters — this is the AUC of all out-of-fold predictions pooled,
    # which is not the same number as the mean of the per-fold AUCs shown in the
    # verdict panel. Labelling both prevents the two from looking contradictory.
    ax.annotate(
        f"{label}\nAUC groupée = {auc:.3f}",
        xy=(0.97, 0.06), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=10, color=INK, fontweight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    style_axes(ax, "Courbe ROC (hors échantillon)", "Taux de faux positifs",
               "Taux de vrais positifs")
    return auc


def plot_fold_auc(ax: Axes, fold_aucs: list[float], base_rate: float | None = None) -> None:
    """Per-fold AUC as a dot plot — the spread is the point.

    A single mean hides that a 5-fold AUC on 132 patients can swing by 0.2. The
    dots make the uncertainty visible instead of implying a precision that the
    cohort size cannot support.
    """
    vals = np.asarray([v for v in fold_aucs if np.isfinite(v)], dtype=float)
    xs = np.arange(1, len(vals) + 1)

    ax.axhline(CHANCE, color=MUTED, linewidth=1.2, linestyle="--", zorder=1)
    ax.annotate("hasard (0.5)", xy=(0.02, CHANCE + 0.012), xycoords=("axes fraction", "data"),
                color=MUTED, fontsize=8)

    mean = float(vals.mean())
    ax.axhline(mean, color=SERIES[0], linewidth=1.6, zorder=2)
    ax.annotate(f"moyenne {mean:.3f}", xy=(0.98, mean + 0.015),
                xycoords=("axes fraction", "data"), ha="right",
                color=INK, fontsize=9, fontweight="bold")

    ax.scatter(xs, vals, s=90, color=SERIES[0], zorder=4,
               edgecolors=SURFACE, linewidths=2.0)
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:.2f}", xy=(x, v), xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)

    ax.set_xticks(xs)
    ax.set_xlim(0.5, len(vals) + 0.5)
    ax.set_ylim(0, 1)
    style_axes(ax, "AUC par pli (patient-wise)", "Pli", "AUC")


def plot_confusion(ax: Axes, y_true: np.ndarray, y_proba: np.ndarray,
                   threshold: float = 0.5) -> None:
    """Confusion matrix with a single-hue sequential ramp (magnitude)."""
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.metrics import confusion_matrix

    pred = (np.asarray(y_proba) >= threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true).astype(int), pred, labels=[0, 1])
    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

    ax.imshow(cm, cmap=cmap, vmin=0, vmax=max(cm.max(), 1))
    labels = ["Non-répondeur", "Répondeur"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Prédit", color=INK_2, fontsize=9)
    ax.set_ylabel("Observé", color=INK_2, fontsize=9)
    ax.set_title("Matrice de confusion (seuil 0,50)", color=INK, fontsize=11,
                 fontweight="bold", loc="left", pad=10)
    ax.grid(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Ink flips to white on the dark end of the ramp so counts stay legible.
    hi = cm.max() or 1
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.annotate(str(val), xy=(j, i), ha="center", va="center",
                        fontsize=15, fontweight="bold",
                        color="#ffffff" if val > 0.55 * hi else INK)
    # A 2px surface gap between cells, per the mark spec.
    for k in (0.5,):
        ax.axhline(k, color=SURFACE, linewidth=2)
        ax.axvline(k, color=SURFACE, linewidth=2)


def plot_calibration(ax: Axes, y_true: np.ndarray, y_proba: np.ndarray,
                     n_bins: int = 5) -> None:
    """Are the predicted probabilities meaningful, or just ranks?"""
    from sklearn.calibration import calibration_curve

    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle="--", zorder=1)
    ax.annotate("parfaitement calibré", xy=(0.30, 0.24), color=MUTED, fontsize=8, rotation=38)
    try:
        frac, mean_pred = calibration_curve(
            np.asarray(y_true).astype(int), y_proba, n_bins=n_bins, strategy="quantile"
        )
        ax.plot(mean_pred, frac, color=SERIES[1], linewidth=2.0,
                marker="o", markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2.0, zorder=3)
        ax.annotate("Modèle", xy=(mean_pred[-1], frac[-1]), xytext=(6, -4),
                    textcoords="offset points", color=INK, fontsize=9,
                    fontweight="bold")
    except ValueError:
        ax.annotate("calibration non calculable\n(trop peu de patients par bin)",
                    xy=(0.5, 0.5), xycoords="axes fraction", ha="center",
                    color=MUTED, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    style_axes(ax, "Courbe de calibration", "Probabilité prédite",
               "Fréquence observée")


# --------------------------------------------------------------------------- #
# Training behaviour and feature attribution
# --------------------------------------------------------------------------- #


def plot_learning_curves(ax: Axes, folds, max_folds: int = 5) -> None:
    """Train vs validation loss. Two series -> legend, in fixed slot order."""
    for i, fold in enumerate(folds[:max_folds]):
        alpha = 0.35 if len(folds) > 1 else 1.0
        ax.plot(fold.train_losses, color=SERIES[0], linewidth=1.6, alpha=alpha,
                label="Entraînement" if i == 0 else None)
        ax.plot(fold.val_losses, color=SERIES[1], linewidth=1.6, alpha=alpha,
                label="Validation" if i == 0 else None)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in leg.get_texts():
        text.set_color(INK_2)
    style_axes(ax, "Courbes de perte (tous les plis)", "Époque d'entraînement",
               "Perte (BCE)")


def permutation_importance(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, names: tuple[str, ...],
    n_repeats: int = 3, top_n: int = 15, seed: int = 0,
) -> tuple[list[str], list[float]]:
    """Drop in AUC when each feature is shuffled, via a fast surrogate model.

    Deliberately *not* the LSTM: permuting 139 features x n_repeats would mean
    hundreds of forward passes for a chart. A logistic surrogate on the
    epoch-averaged features answers the question the chart asks — which inputs
    carry the usable signal — at a fraction of the cost. It is an attribution
    aid, not a claim about the LSTM's internals.
    """
    from sklearn.inspection import permutation_importance as sk_perm
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    flat = x.mean(axis=1)
    y = np.asarray(y).astype(int)
    tr, va = next(GroupKFold(n_splits=5).split(flat, y, groups=groups))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(flat[tr], y[tr])
    res = sk_perm(model, flat[va], y[va], n_repeats=n_repeats,
                  random_state=seed, scoring="roc_auc")

    order = np.argsort(res.importances_mean)[::-1][:top_n]
    return [names[i] for i in order], [float(res.importances_mean[i]) for i in order]


def plot_importance(ax: Axes, names: list[str], values: list[float]) -> None:
    """Horizontal bars — magnitude by length, one hue, largest at the top."""
    ys = np.arange(len(names))[::-1]
    ax.barh(ys, values, color=SERIES[0], height=0.68)
    ax.set_yticks(ys, names)
    ax.axvline(0, color=AXIS, linewidth=1.0)
    # Value labels always sit to the right of the bar's outer edge, including for
    # negative bars: placing a negative label further left runs it straight into
    # the y-axis feature name.
    for yy, val in zip(ys, values):
        ax.annotate(f"{val:+.3f}", xy=(max(val, 0.0), yy),
                    xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8, color=INK_2)
    style_axes(ax, "Importance par permutation",
               "Chute d'AUC si la variable est mélangée", "")
    ax.tick_params(axis="y", labelsize=8)


# --------------------------------------------------------------------------- #
# The verdict — a stat panel, not a chart
# --------------------------------------------------------------------------- #


def verdict(auc: float, auc_std: float, accuracy: float, base_rate: float) -> tuple[str, str]:
    """Classify the result honestly. Returns ``(status, sentence)``.

    Two failure modes are called out explicitly because a bare AUC hides them:
    an accuracy equal to the base rate means the model always predicts the
    majority class, and a fold spread wider than the distance from chance means
    the point estimate is not supported by the data.
    """
    beats_chance = auc - auc_std > CHANCE
    beats_base = accuracy > base_rate + 0.02

    if not beats_chance and not beats_base:
        return "critical", (
            "Au niveau du hasard. L'exactitude n'excède pas le taux de base : "
            "le modèle prédit essentiellement toujours la classe majoritaire."
        )
    if beats_chance and beats_base:
        return "good", (
            "Discrimination utile : l'AUC dépasse le hasard même en tenant compte "
            "de la dispersion entre plis, et l'exactitude dépasse le taux de base."
        )
    if auc > CHANCE and not beats_chance:
        return "warning", (
            "Au-dessus du hasard en moyenne, mais la dispersion entre plis "
            "recouvre 0,5 : le résultat n'est pas établi sur cet effectif."
        )
    return "serious", (
        "Signal marginal — un seul des deux critères (AUC, exactitude) est "
        "satisfait. À interpréter avec prudence."
    )


def plot_verdict(ax: Axes, title: str, auc: float, auc_std: float,
                 accuracy: float, base_rate: float, n_patients: int,
                 n_features: int) -> None:
    """Hero number plus a status word — the panel a reader looks at first."""
    status, sentence = verdict(auc, auc_std, accuracy, base_rate)
    colour = STATUS[status]
    icon = {"good": "✓", "warning": "!", "serious": "!", "critical": "✕"}[status]
    word = {"good": "SIGNAL UTILE", "warning": "NON ÉTABLI",
            "serious": "MARGINAL", "critical": "AU NIVEAU DU HASARD"}[status]

    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    ax.annotate(title, xy=(0.0, 0.97), xycoords="axes fraction",
                fontsize=11, fontweight="bold", color=INK, va="top")
    ax.annotate(f"{auc:.3f}", xy=(0.0, 0.84), xycoords="axes fraction",
                fontsize=40, fontweight="bold", color=INK, va="top")
    # Sits below the hero number, not beside it: at 40pt the digits are wide
    # enough to collide with anything placed on the same line.
    ax.annotate(f"AUC moyenne par pli  ± {auc_std:.3f}",
                xy=(0.0, 0.53), xycoords="axes fraction",
                fontsize=9.5, color=INK_2, va="top")

    # Status: colour + icon + word, never colour alone.
    ax.annotate(f"{icon}  {word}", xy=(0.0, 0.42), xycoords="axes fraction",
                fontsize=12, fontweight="bold", color=colour, va="top")
    # matplotlib's wrap=True does not respect axes bounds for annotations, so the
    # sentence is wrapped explicitly — otherwise it runs into the next panel.
    ax.annotate("\n".join(textwrap.wrap(sentence, width=52)),
                xy=(0.0, 0.31), xycoords="axes fraction",
                fontsize=8.5, color=INK_2, va="top", linespacing=1.5)
    ax.annotate(
        f"{n_patients} patients · {n_features} variables\n"
        f"exactitude {accuracy:.3f} vs taux de base {base_rate:.3f}",
        xy=(0.0, 0.02), xycoords="axes fraction", fontsize=8, color=MUTED,
        va="bottom", linespacing=1.6,
    )


def model_report(
    name: str, cv_result, y: np.ndarray, x: np.ndarray, groups: np.ndarray,
    names: tuple[str, ...], base_rate: float,
) -> Figure:
    """The six-panel report every notebook ends with."""
    import matplotlib.pyplot as plt

    summary = cv_result.summary()
    y_true, y_proba = cv_result.out_of_fold(y)

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.0), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)

    plot_verdict(axes[0, 0], name, summary["auc_mean"], summary["auc_std"],
                 summary["accuracy_mean"], base_rate, len(y), x.shape[-1])
    plot_roc(axes[0, 1], y_true, y_proba)
    plot_fold_auc(axes[0, 2], [f.auc for f in cv_result.folds])
    plot_confusion(axes[1, 0], y_true, y_proba)
    plot_calibration(axes[1, 1], y_true, y_proba)

    feats, vals = permutation_importance(x, y, groups, names, top_n=12)
    plot_importance(axes[1, 2], feats, vals)

    return fig
