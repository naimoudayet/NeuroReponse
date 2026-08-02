from __future__ import annotations

import io
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session, sessionmaker

from ..domain.patient import DossierClinique, Patient
from ..domain.prediction import Prediction
from ..domain.rtms_parameters import RTMSParameters
from ..domain.session_rtms import SessionRTMS
from ..domain.signal_neuro import SignalNeurophysiologique, SignalType
from .schema import Base, PatientRow, PredictionRow, SessionRow, SignalRow


def _npy_dumps(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _npy_loads(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


class Repository:
    def __init__(self, db_url: str = "sqlite:///recherche.sqlite3") -> None:
        self.engine = create_engine(db_url, future=True)
        Base.metadata.create_all(self.engine)
        self._Session: sessionmaker[Session] = sessionmaker(bind=self.engine, expire_on_commit=False)

    def sauvegarder_patient(self, patient: Patient) -> None:
        with self._Session.begin() as s:
            row = s.get(PatientRow, patient.id) or PatientRow(id=patient.id)
            row.nom = patient.nom
            row.age = patient.age
            row.diagnostic = patient.diagnostic
            row.sexe = patient.sexe
            row.historique_json = json.dumps(
                [
                    {"date": d.date.isoformat(), "note": d.note, "score_depression": d.score_depression}
                    for d in patient.historique_clinique
                ]
            )
            s.merge(row)

            for sess in patient.sessions:
                self._upsert_session(s, sess)

    def charger_patient(self, patient_id: str) -> Patient | None:
        with self._Session() as s:
            row = s.get(PatientRow, patient_id)
            if row is None:
                return None
            historique = [
                DossierClinique(
                    date=datetime.fromisoformat(d["date"]),
                    note=d["note"],
                    score_depression=d.get("score_depression"),
                )
                for d in json.loads(row.historique_json)
            ]
            sessions = [self._row_to_session(sr, s) for sr in row.sessions]
            return Patient(
                id=row.id,
                nom=row.nom,
                age=row.age,
                diagnostic=row.diagnostic,
                sexe=row.sexe,
                historique_clinique=historique,
                sessions=sessions,
            )

    def rechercher_session(self, id_session: str) -> SessionRTMS | None:
        with self._Session() as s:
            row = s.get(SessionRow, id_session)
            return self._row_to_session(row, s) if row else None

    def lister_sessions_patient(self, patient_id: str) -> list[SessionRTMS]:
        with self._Session() as s:
            rows = s.execute(select(SessionRow).where(SessionRow.patient_id == patient_id)).scalars().all()
            return [self._row_to_session(r, s) for r in rows]

    def sauvegarder_prediction(self, prediction: Prediction) -> int:
        with self._Session.begin() as s:
            row = PredictionRow(
                patient_id=prediction.patient_id,
                valeur=prediction.valeur,
                probabilite=prediction.probabilite,
                date=prediction.date,
                type=prediction.type,
                model_version=prediction.model_version,
            )
            s.add(row)
            s.flush()
            return row.id

    def lister_predictions(self, patient_id: str) -> list[Prediction]:
        with self._Session() as s:
            rows = s.execute(select(PredictionRow).where(PredictionRow.patient_id == patient_id)).scalars().all()
            return [
                Prediction(
                    patient_id=r.patient_id,
                    valeur=r.valeur,
                    probabilite=r.probabilite,
                    date=r.date,
                    type=r.type,
                    model_version=r.model_version,
                )
                for r in rows
            ]

    def charger_modele(self, version: str) -> Path | None:
        from .schema import ModelRow

        with self._Session() as s:
            row = s.get(ModelRow, version)
            return Path(row.weights_path) if row else None

    def _upsert_session(self, s: Session, sess: SessionRTMS) -> None:
        row = s.get(SessionRow, sess.id_session) or SessionRow(id_session=sess.id_session)
        row.patient_id = sess.patient_id
        row.date = sess.date
        row.statut = sess.statut
        row.score_pre = sess.score_pre
        row.score_post = sess.score_post
        row.parametres_json = json.dumps(asdict(sess.parametres))
        s.merge(row)

        # Replace, don't append: without this a second save of the same session
        # stacks a duplicate copy of every channel, which silently corrupts the
        # montage shape the model is fed at inference. Sessions saved without
        # signals leave the stored ones untouched (nothing to write).
        if sess.signaux:
            s.execute(delete(SignalRow).where(SignalRow.session_id == sess.id_session))

        for sig in sess.signaux:
            s.add(
                SignalRow(
                    session_id=sess.id_session,
                    type_signal=sig.type_signal.value,
                    canal=sig.canal,
                    sampling_rate_hz=sig.sampling_rate_hz,
                    timestamp=sig.timestamp,
                    valeurs_npy=_npy_dumps(sig.valeurs),
                )
            )

    def _row_to_session(self, row: SessionRow, s: Session) -> SessionRTMS:
        params_dict = json.loads(row.parametres_json)
        params = RTMSParameters(**params_dict)
        signaux = [
            SignalNeurophysiologique(
                type_signal=SignalType(sig.type_signal),
                valeurs=_npy_loads(sig.valeurs_npy),
                timestamp=sig.timestamp,
                canal=sig.canal,
                sampling_rate_hz=sig.sampling_rate_hz,
            )
            for sig in row.signaux
        ]
        return SessionRTMS(
            id_session=row.id_session,
            patient_id=row.patient_id,
            parametres=params,
            date=row.date,
            signaux=signaux,
            score_pre=row.score_pre,
            score_post=row.score_post,
            statut=row.statut,
        )
