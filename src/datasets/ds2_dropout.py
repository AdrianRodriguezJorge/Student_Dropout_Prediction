# ─────────────────────────────────────────────────────────────────────────────
#  src/datasets/ds2_dropout.py
#  Predict Student Dropout — UCI / SATDAP, 2022
#  Target: Dropout=1, Graduate or Enrolled=0
#
#  "Enrolled" students are treated as non-dropout because they have not yet
#  abandoned their program; they are still active.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
from typing import Tuple

from src.datasets.base import DatasetConfig

_PATH   = "data/dropout.csv"
_TARGET = "Target"

# All feature columns are already numeric (coded integers or floats).
# No nominal string columns exist — no OHE needed.
_NUMERIC_COLS = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification",
    "Previous qualification (grade)", "Nacionality",
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
    "Admission grade", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder",
    "Age at enrollment", "International",
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP",
]


def _load() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(_PATH, sep=";")
    df.columns = [c.strip() for c in df.columns]

    # Binary target: Dropout → 1, Graduate / Enrolled → 0
    y = (df[_TARGET] == "Dropout").astype(int)
    X = df[_NUMERIC_COLS].copy()
    return X, y


def _slider(col: str, label: str, lo, hi, default) -> dict:
    return {"col": col, "label": label, "type": "slider",
            "min": lo, "max": hi, "default": default}


_UI_FORM = [
    _slider("Age at enrollment",           "Edad al inscribirse",         15, 70, 20),
    _slider("Application mode",            "Modalidad de aplicacion",      1, 57,  1),
    _slider("Application order",           "Orden de preferencia",         0,  9,  1),
    _slider("Previous qualification",      "Calificacion previa",          1, 43,  1),
    _slider("Previous qualification (grade)", "Nota calificacion previa", 95, 200, 130),
    _slider("Admission grade",             "Nota de admision",            95, 200, 130),
    _slider("Curricular units 1st sem (enrolled)",  "Materias inscritas 1S", 0, 26, 6),
    _slider("Curricular units 1st sem (approved)",  "Materias aprobadas 1S", 0, 26, 5),
    _slider("Curricular units 1st sem (grade)",     "Nota promedio 1S",      0, 20, 12),
    _slider("Curricular units 2nd sem (enrolled)",  "Materias inscritas 2S", 0, 23, 6),
    _slider("Curricular units 2nd sem (approved)",  "Materias aprobadas 2S", 0, 20, 5),
    _slider("Curricular units 2nd sem (grade)",     "Nota promedio 2S",      0, 20, 12),
    _slider("Tuition fees up to date",     "Colegiatura al dia (0/1)",     0,  1,  1),
    _slider("Scholarship holder",          "Tiene beca (0/1)",             0,  1,  0),
    _slider("Debtor",                      "Deudor (0/1)",                 0,  1,  0),
    _slider("Displaced",                   "Desplazado (0/1)",             0,  1,  0),
    _slider("Gender",                      "Genero (0=F, 1=M)",            0,  1,  0),
    _slider("Unemployment rate",           "Tasa de desempleo (%)",        7, 17, 11),
    _slider("Inflation rate",              "Tasa de inflacion (%)",       -1,  3,  1),
    _slider("GDP",                         "PIB",                         -4,  4,  1),
]


config = DatasetConfig(
    name          = "Prediccion de Abandono Universitario",
    description   = "Predice si un estudiante abandonara su carrera. Portugal, 2022.",
    load_fn       = _load,
    nominal_cols  = [],
    numeric_cols  = _NUMERIC_COLS,
    target_labels = ("No abandona", "Abandona"),
    ui_form       = _UI_FORM,
)
