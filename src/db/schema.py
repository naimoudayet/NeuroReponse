from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey, LargeBinary, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class PatientRow(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    nom: Mapped[str] = mapped_column(String)
    age: Mapped[float]
    diagnostic: Mapped[str] = mapped_column(String)
    # Nullable : le simulateur séquentiel historique ne publie pas cette variable.
    sexe: Mapped[int | None] = mapped_column(default=None)
    historique_json: Mapped[str] = mapped_column(String, default="[]")

    # Ordered, always. The sequence axis of the LSTM *is* this list: the model
    # eats sessions[0..n] as timesteps, `etapes_boucle` reads the change from one
    # to the next, and `analyser_suivi` fits a slope over them. SQLite returns
    # rows in rowid order absent an ORDER BY, which happens to match the seeders'
    # insertion order and stops matching the moment a missed visit is recorded
    # late. That reorders the treatment course silently — no error, just wrong
    # predictions — so the order is pinned here rather than left to chance.
    # Date first (the clinical truth), id as tiebreaker for cohorts whose
    # "sessions" are epochs of one recording and therefore share a timestamp.
    sessions: Mapped[list[SessionRow]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
        order_by="(SessionRow.date, SessionRow.id_session)",
    )


class SessionRow(Base):
    __tablename__ = "sessions"

    id_session: Mapped[str] = mapped_column(String, primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"))
    date: Mapped[datetime]
    statut: Mapped[str] = mapped_column(String, default="planifiee")
    score_pre: Mapped[float | None]
    score_post: Mapped[float | None]
    parametres_json: Mapped[str] = mapped_column(String)

    patient: Mapped[PatientRow] = relationship(back_populates="sessions")
    signaux: Mapped[list[SignalRow]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class SignalRow(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id_session"))
    type_signal: Mapped[str] = mapped_column(String)
    canal: Mapped[str] = mapped_column(String)
    sampling_rate_hz: Mapped[float]
    timestamp: Mapped[datetime]
    valeurs_npy: Mapped[bytes] = mapped_column(LargeBinary)

    session: Mapped[SessionRow] = relationship(back_populates="signaux")


class PredictionRow(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"))
    valeur: Mapped[int]
    probabilite: Mapped[float]
    date: Mapped[datetime]
    type: Mapped[str] = mapped_column(String, default="responder_classification")
    model_version: Mapped[str | None]


class ModelRow(Base):
    __tablename__ = "models"

    version: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime]
    architecture_json: Mapped[str] = mapped_column(String)
    metrics_json: Mapped[str] = mapped_column(String, default="{}")
    weights_path: Mapped[str] = mapped_column(String)
