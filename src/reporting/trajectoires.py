"""Export how the trained model reacts to a patient across successive sessions.

    python -m src.reporting.trajectoires

Writes ``docs/strategy/trajectoires_seances.csv`` (one row per patient × session),
``docs/strategy/trajectoires_resume.csv`` (mean TRI per session, split by observed
outcome) and ``docs/strategy/trajectoires_seances.png``.

**Only the sequential cohort can answer this question.** It is the one place in
this project where the sequence axis is a real treatment course, so
``predict_tri()[k]`` genuinely means "the model's estimate after session k+1" and
the trajectory is the LSTM accumulating evidence. On TDBRAIN the same call would
return one value per *epoch of a single baseline recording*, which is not a
course and must not be plotted as one — see :mod:`src.reporting.suivi`.

Inference goes through :func:`src.app.inference.build_model_input`, the same
function the Predictions and Suivi pages call. Rebuilding the tensor here would
let this export drift away from what the application shows for the same patient.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from ..app.inference import build_model_input
from ..app.utils import SOURCES, DataSource
from ..db import Repository
from ..models.train import load_model
from .suivi import SEUIL_REPONSE

OUT_DIR = Path("docs/strategy")
DETAIL = OUT_DIR / "trajectoires_seances.csv"
RESUME = OUT_DIR / "trajectoires_resume.csv"
FIGURE = OUT_DIR / "trajectoires_seances.png"

FIELDS = [
    "patient_id", "repondeur_observe", "reduction_observee",
    "seance", "id_session", "date",
    "score_pre", "score_post", "tri", "delta_tri",
]


def _observed(patient) -> tuple[float | None, int | None]:
    """Observed score reduction, and the **ground-truth** responder label.

    The label is read from the clinical record, not re-derived from a 50 %
    reduction rule. That distinction is load-bearing on this cohort: the
    simulator draws a 50/50 split (``SimConfig.responder_rate``) and moves a
    responder's score by only 0.5–1.5 points per session, so ten sessions from a
    baseline of 20–30 yield roughly a 25–50 % reduction. Applying the standard
    BDI-II criterion here labels almost everyone a non-responder (measured: 3 of
    100) and silently destroys the comparison — the group means would then rest
    on three patients.

    ``src.data.seeder`` writes the true label into the second clinical entry as
    "Répondeur"/"Non-répondeur (label simulé)"; that note is the only place the
    simulator's ground truth survives into SQLite.
    """
    pre = patient.sessions[0].score_pre
    post = patient.sessions[-1].score_post
    reduction = (
        None if pre is None or post is None or pre <= 0 else (pre - post) / pre
    )

    for dossier in patient.historique_clinique:
        note = (dossier.note or "").strip().lower()
        if note.startswith("non-répondeur"):
            return reduction, 0
        if note.startswith("répondeur"):
            return reduction, 1

    # No stored label (a hand-created patient): fall back to the clinical rule.
    if reduction is None:
        return None, None
    return reduction, int(reduction >= SEUIL_REPONSE)


def collect(limit: int | None = None) -> list[dict]:
    """One row per patient × session, with the model's running estimate."""
    cfg = SOURCES[DataSource.SIMULE_SEQ]
    if not cfg.db.exists():
        raise SystemExit(
            f"{cfg.db} absent — seed the sequential cohort first "
            f"(`python -m src.data.seeder`)."
        )
    model_path = cfg.models[0].model
    if not model_path.exists():
        raise SystemExit(f"{model_path} absent — train the sequential model first.")

    repo = Repository(db_url=f"sqlite:///{cfg.db}")
    model = load_model(model_path)

    from sqlalchemy import select

    from ..db.schema import PatientRow
    with repo._Session() as s:
        ids = list(s.execute(select(PatientRow.id).order_by(PatientRow.id)).scalars())
    if limit:
        ids = ids[:limit]

    rows: list[dict] = []
    for pid in ids:
        patient = repo.charger_patient(pid)
        if patient is None or not patient.sessions:
            continue
        if not all(sess.signaux for sess in patient.sessions):
            continue

        x, _fs = build_model_input(patient, is_real=False)
        with torch.no_grad():
            tri = model.predict_tri(
                torch.as_tensor(x, dtype=torch.float32)
            ).squeeze(0).cpu().numpy().tolist()

        reduction, responder = _observed(patient)
        prev: float | None = None
        for i, sess in enumerate(patient.sessions):
            val = float(tri[i]) if i < len(tri) else None
            rows.append({
                "patient_id": pid,
                "repondeur_observe": responder,
                "reduction_observee": None if reduction is None else round(reduction, 4),
                "seance": i + 1,
                "id_session": sess.id_session,
                "date": sess.date.date().isoformat(),
                "score_pre": None if sess.score_pre is None else round(sess.score_pre, 2),
                "score_post": None if sess.score_post is None else round(sess.score_post, 2),
                "tri": None if val is None else round(val, 4),
                "delta_tri": None if (val is None or prev is None) else round(val - prev, 4),
            })
            if val is not None:
                prev = val
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    """Mean TRI per session, split by the *observed* outcome.

    This is the table that answers the question directly: if the model is
    accumulating evidence, the two groups separate as sessions go by.
    """
    seances = sorted({r["seance"] for r in rows})
    out: list[dict] = []
    for k in seances:
        at_k = [r for r in rows if r["seance"] == k and r["tri"] is not None]
        rep = [r["tri"] for r in at_k if r["repondeur_observe"] == 1]
        non = [r["tri"] for r in at_k if r["repondeur_observe"] == 0]
        out.append({
            "seance": k,
            "n_repondeurs": len(rep),
            "n_non_repondeurs": len(non),
            "tri_moyen_repondeurs": round(float(np.mean(rep)), 4) if rep else None,
            "tri_moyen_non_repondeurs": round(float(np.mean(non)), 4) if non else None,
            "ecart": (round(float(np.mean(rep)) - float(np.mean(non)), 4)
                      if rep and non else None),
        })
    return out


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _figure(rows: list[dict], resume: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: every patient's trajectory, coloured by observed outcome.
    by_patient: dict[str, list[tuple[int, float]]] = {}
    outcome: dict[str, int | None] = {}
    for r in rows:
        if r["tri"] is None:
            continue
        by_patient.setdefault(r["patient_id"], []).append((r["seance"], r["tri"]))
        outcome[r["patient_id"]] = r["repondeur_observe"]
    for pid, pts in by_patient.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        col = "#0089A6" if outcome[pid] == 1 else "#B4621F"
        ax1.plot(xs, ys, color=col, alpha=0.18, linewidth=1)
    ax1.axhline(SEUIL_REPONSE, ls="--", color="#666", lw=1)
    ax1.set_title("Trajectoire du modèle, patient par patient")
    ax1.set_xlabel("Séance")
    ax1.set_ylabel("TRI — P(réponse)")
    ax1.set_ylim(0, 1)

    # Right: the group means — the separation is the point.
    ks = [r["seance"] for r in resume]
    ax2.plot(ks, [r["tri_moyen_repondeurs"] for r in resume],
             "o-", color="#0089A6", lw=2.5, label="Répondeurs (observé)")
    ax2.plot(ks, [r["tri_moyen_non_repondeurs"] for r in resume],
             "s-", color="#B4621F", lw=2.5, label="Non-répondeurs (observé)")
    ax2.axhline(SEUIL_REPONSE, ls="--", color="#666", lw=1)
    ax2.set_title("TRI moyen par séance, selon l'issue réelle")
    ax2.set_xlabel("Séance")
    ax2.set_ylabel("TRI moyen")
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=False)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)


def build(limit: int | None = None) -> tuple[Path, Path, Path]:
    rows = collect(limit)
    resume = summarise(rows)
    _write(DETAIL, FIELDS, rows)
    _write(RESUME, list(resume[0].keys()), resume)
    _figure(rows, resume)

    print(f"{len({r['patient_id'] for r in rows})} patients · {len(rows)} lignes\n")
    print(f"{'séance':>7} {'TRI répondeurs':>15} {'TRI non-rép.':>13} {'écart':>8}")
    for r in resume:
        print(f"{r['seance']:>7} {r['tri_moyen_repondeurs']:>15.3f} "
              f"{r['tri_moyen_non_repondeurs']:>13.3f} {r['ecart']:>8.3f}")
    return DETAIL, RESUME, FIGURE


if __name__ == "__main__":
    d, r, f = build()
    print(f"\nécrit : {d}\n        {r}\n        {f}")
