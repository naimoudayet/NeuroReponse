"""Build the user-facing DOCX guide.

Run after capture_screens.py has produced docs/screenshots/*.png:
    python -m src.reporting.user_guide
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

SCREENSHOTS = Path("docs/screenshots")
OUTPUT = Path("docs/Guide_Utilisateur.docx")


def _heading(doc: Document, text: str, level: int = 1) -> None:
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)


def _step(doc: Document, n: int, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(f"Étape {n}. ")
    r.bold = True
    p.add_run(text)


def _note(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run("Note : ")
    r.bold = True
    r.font.color.rgb = RGBColor(0xB4, 0x53, 0x09)
    p.add_run(text)


def _figure(doc: Document, image: Path, caption: str, width_cm: float = 15.0) -> None:
    if not image.exists():
        doc.add_paragraph(f"[capture manquante : {image.name}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(image), width=Cm(width_cm))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Figure : {caption}")
    r.italic = True
    r.font.size = Pt(9)


def _code(doc: Document, lines: list[str]) -> None:
    for line in lines:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.5)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(line)
        r.font.name = "Consolas"
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)


def build() -> Path:
    doc = Document()

    # ------------------------------------------------------------------ Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t = title.add_run("Guide d'utilisation")
    t.bold = True
    t.font.size = Pt(28)
    t.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s = sub.add_run("Application Recherche-App — rTMS + LSTM")
    s.font.size = Pt(14)

    sub2 = doc.add_paragraph()
    sub2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s2 = sub2.add_run("PFE 2026")
    s2.italic = True
    s2.font.size = Pt(11)

    doc.add_paragraph()
    intro = doc.add_paragraph()
    intro.add_run(
        "Ce guide explique pas-à-pas comment installer, lancer et utiliser "
        "l'application de prédiction de la réponse au traitement rTMS basée sur un "
        "modèle LSTM. Il s'adresse à un utilisateur (clinicien, chercheur ou "
        "examinateur) qui découvre l'application pour la première fois."
    )

    doc.add_page_break()

    # ------------------------------------------------------------------ Install
    _heading(doc, "1. Installation et démarrage")

    doc.add_paragraph(
        "L'application est écrite en Python et utilise PyTorch (CPU) pour le modèle "
        "LSTM, SQLite pour le stockage et Streamlit pour l'interface graphique."
    )

    _heading(doc, "1.1 Prérequis", level=2)
    doc.add_paragraph("• Python 3.11 ou plus récent (testé sur 3.14).")
    doc.add_paragraph("• Environ 500 Mo d'espace disque (PyTorch + dépendances).")
    doc.add_paragraph("• Un navigateur web (l'app s'ouvre dans le navigateur).")

    _heading(doc, "1.2 Installation", level=2)
    _step(doc, 1, "Ouvrir une console PowerShell dans le dossier du projet.")
    _step(doc, 2, "Créer et activer un environnement virtuel :")
    _code(doc, ["python -m venv .venv", ".\\.venv\\Scripts\\Activate.ps1"])
    _step(doc, 3, "Installer les dépendances :")
    _code(doc, [
        "pip install -r requirements.txt",
        "pip install torch --index-url https://download.pytorch.org/whl/cpu",
    ])

    _heading(doc, "1.3 Lancer l'application", level=2)
    _step(doc, 1, "Démarrer le serveur Streamlit :")
    _code(doc, ["streamlit run src/app/main.py"])
    _step(doc, 2, "Le navigateur s'ouvre automatiquement sur http://localhost:8501.")

    doc.add_page_break()

    # ------------------------------------------------------------------ Home
    _heading(doc, "2. Page d'accueil")
    doc.add_paragraph(
        "La page d'accueil affiche le statut global : nombre de patients en base, "
        "présence d'un modèle entraîné, et backend utilisé."
    )
    _figure(doc, SCREENSHOTS / "01_home.png", "Page d'accueil — vue initiale.")
    _step(doc, 1,
          "Si la base est vide, cliquer sur « Initialiser avec les données simulées ». "
          "L'opération génère 100 patients fictifs avec 10 séances rTMS chacun.")
    _step(doc, 2,
          "Une fois la base initialisée, utiliser la barre latérale pour naviguer "
          "entre les pages : Patients, Sessions, Training, Predictions.")
    _note(doc,
          "L'initialisation prend quelques secondes. Recharge la page après pour voir "
          "les compteurs mis à jour.")

    doc.add_page_break()

    # ------------------------------------------------------------------ Patients
    _heading(doc, "3. Page Patients")
    doc.add_paragraph(
        "Cette page liste tous les patients enregistrés et permet de consulter leur "
        "dossier, d'ajouter une entrée clinique ou de créer un nouveau patient."
    )
    _figure(doc, SCREENSHOTS / "02_patients.png", "Page Patients — liste et détail.")
    _step(doc, 1, "Consulter le tableau récapitulatif en haut (ID, nom, âge, diagnostic, etc.).")
    _step(doc, 2, "Sélectionner un patient dans la liste déroulante « Patient » "
          "pour afficher son détail et son historique clinique.")
    _step(doc, 3, "Pour ajouter une note ou un score de dépression à un patient : "
          "déplier la section « Ajouter une entrée au dossier », remplir le formulaire, "
          "puis cliquer sur « Ajouter ».")
    _step(doc, 4, "Pour créer un nouveau patient : déplier « Créer un nouveau patient », "
          "saisir un ID unique, un nom, un âge et un diagnostic, puis cliquer sur « Créer ».")

    doc.add_page_break()

    # ------------------------------------------------------------------ Sessions
    _heading(doc, "4. Page Sessions")
    doc.add_paragraph(
        "Cette page permet de visualiser les séances rTMS d'un patient : les paramètres "
        "de stimulation (fréquence, intensité, localisation), le signal EEG enregistré "
        "et les features extraits."
    )
    _figure(doc, SCREENSHOTS / "03_sessions.png", "Page Sessions — visualisation d'un signal EEG.")
    _step(doc, 1, "Sélectionner le patient puis la séance à visualiser.")
    _step(doc, 2, "Lire les indicateurs en haut : statut de la séance, score pré/post, "
          "nombre de signaux enregistrés.")
    _step(doc, 3, "Vérifier les paramètres rTMS (fréquence, intensité, durée des trains, "
          "localisation, nombre total d'impulsions calculé automatiquement).")
    _step(doc, 4, "Choisir un canal dans la liste pour afficher le tracé du signal EEG "
          "dans le temps. Le tracé est interactif (zoom, pan).")
    _step(doc, 5, "Déplier « Features extraits » pour voir les statistiques temporelles "
          "(moyenne, RMS) et les puissances dans chaque bande de fréquence (delta, "
          "theta, alpha, beta, gamma).")
    _step(doc, 6, "Déplier « Rapport de séance » pour obtenir un résumé JSON exportable.")

    doc.add_page_break()

    # ------------------------------------------------------------------ Training
    _heading(doc, "5. Page Training (entraînement du modèle)")
    doc.add_paragraph(
        "Cette page entraîne le modèle LSTM sur les données du jeu de données simulé "
        "et évalue sa performance par validation croisée patient-wise (un patient "
        "n'apparaît jamais à la fois en train et en validation)."
    )
    _figure(doc, SCREENSHOTS / "04_training.png", "Page Training — hyperparamètres et validation croisée.")
    _step(doc, 1, "Régler les hyperparamètres si besoin (les valeurs par défaut sont "
          "raisonnables) : nombre d'epochs, batch size, learning rate, nombre de folds, "
          "dropout, seed.")
    _step(doc, 2, "Onglet « Validation croisée » : cliquer sur « Lancer la validation "
          "croisée patient-wise ». L'entraînement dure environ 1 à 3 minutes (CPU).")
    _step(doc, 3, "Une fois terminé, lire les trois indicateurs en haut : Accuracy, AUC, "
          "F1 (moyenne ± écart-type sur les folds).")
    _step(doc, 4, "Examiner les courbes de loss par fold pour repérer un éventuel "
          "surapprentissage (train ↓ mais val ↑).")
    _step(doc, 5, "Onglet « Modèle final » : cliquer sur « Entraîner et sauvegarder le "
          "modèle final ». Le modèle est sauvegardé dans `data/models/lstm_v1.pt` et "
          "sera utilisé par la page Predictions.")
    _note(doc, "Les chiffres élevés sur les données simulées (AUC ≈ 0,99) sont attendus : "
          "le simulateur injecte délibérément un biomarqueur appris par le modèle. "
          "Sur des données réelles, l'AUC se situerait plutôt entre 0,6 et 0,75.")

    doc.add_page_break()

    # ------------------------------------------------------------------ Predictions
    _heading(doc, "6. Page Predictions")
    doc.add_paragraph(
        "Cette page utilise le modèle entraîné pour prédire la probabilité qu'un "
        "patient soit répondeur au traitement rTMS, à partir de ses séances enregistrées."
    )
    _figure(doc, SCREENSHOTS / "05_predictions.png", "Page Predictions — résultat pour un patient.")
    _step(doc, 1, "Sélectionner un patient. Le modèle est appliqué automatiquement "
          "à l'ensemble de ses séances.")
    _step(doc, 2, "Lire les trois indicateurs : probabilité de réponse, classe prédite "
          "(Répondeur / Non-répondeur), nombre de séances utilisées.")
    _step(doc, 3, "L'interprétation textuelle est affichée juste en dessous : "
          "« Répondeur (XX %) — patient PNNN ».")
    _step(doc, 4, "Si le patient a un score clinique observé, l'écart entre la prédiction "
          "et l'observation est calculé automatiquement.")
    _step(doc, 5, "Cliquer sur « Sauvegarder cette prédiction en base » pour conserver "
          "l'historique. Les prédictions précédentes du patient sont affichées dans un "
          "tableau juste en dessous.")
    _step(doc, 6, "Cliquer sur « Télécharger le rapport PDF » pour exporter un PDF de "
          "la prédiction (pour le dossier patient ou la communication au clinicien).")
    _note(doc, "Le modèle doit être entraîné avant d'utiliser cette page. Si le message "
          "« Aucun modèle entraîné » apparaît, retourner sur la page Training.")

    doc.add_page_break()

    # ------------------------------------------------------------------ Pipeline summary
    _heading(doc, "7. Workflow complet (pour mémoire)")
    doc.add_paragraph(
        "Un cycle d'utilisation complet, de la première installation à l'export d'un "
        "rapport PDF, suit cet enchaînement :"
    )

    workflow = [
        "Accueil → cliquer sur « Initialiser avec les données simulées ».",
        "Patients → vérifier qu'il y a 100 patients dans le tableau.",
        "Sessions → choisir un patient, visualiser une séance et son EEG.",
        "Training → onglet « Validation croisée » → lancer l'entraînement.",
        "Training → onglet « Modèle final » → sauvegarder les poids.",
        "Predictions → choisir un patient → lire la prédiction → exporter PDF.",
    ]
    for i, step in enumerate(workflow, 1):
        _step(doc, i, step)

    _heading(doc, "8. En cas de problème", level=1)
    troubles = [
        ("La page Predictions affiche « Aucun modèle entraîné »",
         "Aller sur la page Training et cliquer sur « Entraîner et sauvegarder le modèle final »."),
        ("Le message « Base de données vide » s'affiche en boucle",
         "Lancer manuellement `python -m src.data.seeder` depuis la console."),
        ("L'installation de TensorFlow échoue",
         "C'est normal sur Python 3.14. L'application utilise PyTorch — voir la commande "
         "d'installation dans la section 1.2."),
        ("Le port 8501 est déjà utilisé",
         "Lancer Streamlit sur un autre port : `streamlit run src/app/main.py --server.port 8502`."),
    ]
    for problem, solution in troubles:
        p = doc.add_paragraph()
        r = p.add_run(f"• {problem}")
        r.bold = True
        doc.add_paragraph(f"  → {solution}")

    # ---- Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(f"  wrote {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    build()
