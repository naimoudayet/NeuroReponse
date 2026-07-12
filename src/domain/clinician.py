from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .prediction import Prediction
from .rtms_parameters import RTMSParameters


class Role(str, Enum):
    CLINICIEN = "clinicien"
    CHERCHEUR = "chercheur"
    ADMIN = "admin"


@dataclass
class ClinicianInterface:
    nom_utilisateur: str
    role: Role = Role.CLINICIEN

    def afficher_resultats(self, predictions: list[Prediction]) -> list[str]:
        return [p.afficher() for p in predictions]

    def modifier_parametres(self, params: RTMSParameters, **updates) -> RTMSParameters:
        for key, value in updates.items():
            if not hasattr(params, key):
                raise AttributeError(f"RTMSParameters has no field '{key}'")
            setattr(params, key, value)
        return params

    def exporter_pdf(self, predictions: list[Prediction], output) -> None:
        """Write a one-page PDF report. `output` is a file path or a file-like object."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(output, pagesize=A4)
        width, height = A4
        y = height - 2 * cm

        c.setFont("Helvetica-Bold", 16)
        c.drawString(2 * cm, y, "Rapport de prédiction rTMS + LSTM")
        y -= 1.2 * cm
        c.setFont("Helvetica", 10)
        c.drawString(2 * cm, y, f"Clinicien : {self.nom_utilisateur}  —  Rôle : {self.role.value}")
        y -= 0.6 * cm
        from datetime import datetime
        c.drawString(2 * cm, y, f"Généré le : {datetime.now().isoformat(timespec='seconds')}")
        y -= 1 * cm

        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, y, f"Prédictions ({len(predictions)})")
        y -= 0.8 * cm

        c.setFont("Helvetica", 10)
        for p in predictions:
            line = (
                f"- Patient {p.patient_id} | classe={'Répondeur' if p.valeur == 1 else 'Non-répondeur'}"
                f" | proba={p.probabilite:.1%} | modèle={p.model_version or 'n/a'}"
                f" | {p.date.isoformat(timespec='seconds')}"
            )
            c.drawString(2 * cm, y, line)
            y -= 0.55 * cm
            if y < 2 * cm:
                c.showPage()
                y = height - 2 * cm
                c.setFont("Helvetica", 10)

        c.showPage()
        c.save()
