# ─────────────────────────────────────────────────────────────────────────────
#  src/datasets/ds3_higher_ed.py
#  Higher Education Students Performance Evaluation — Yilmaz & Sekeroglu, 2019
#  Target: GRADE == 0 → Fail (0),  GRADE >= 1 → Pass (1)
#
#  All 30 survey features are already coded as integers (1-based categories).
#  STUDENT ID and COURSE ID are identifiers — excluded.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
from typing import Tuple

from src.datasets.base import DatasetConfig

_PATH      = "data/higher_ed.csv"
_TARGET    = "GRADE"
_EXCLUDED  = ["STUDENT ID", "COURSE ID"]

# Columns are named "1" through "30" in the CSV
_FEATURE_COLS = [str(i) for i in range(1, 31)]

# Human-readable labels for each numbered column
_COL_LABELS = {
    "1":  "Edad",
    "2":  "Sexo",
    "3":  "Tipo bachillerato",
    "4":  "Tipo beca",
    "5":  "Trabajo adicional",
    "6":  "Actividad artistica/deportiva",
    "7":  "Tiene pareja",
    "8":  "Salario total",
    "9":  "Transporte",
    "10": "Alojamiento",
    "11": "Educacion madre",
    "12": "Educacion padre",
    "13": "Numero hermanos",
    "14": "Estado parental",
    "15": "Ocupacion madre",
    "16": "Ocupacion padre",
    "17": "Horas estudio semanales",
    "18": "Frecuencia lectura (no cientifica)",
    "19": "Frecuencia lectura (cientifica)",
    "20": "Asistencia seminarios",
    "21": "Impacto proyectos en exito",
    "22": "Asistencia a clases",
    "23": "Preparacion parciales (modo)",
    "24": "Preparacion parciales (momento)",
    "25": "Toma notas",
    "26": "Atencion en clases",
    "27": "Discusion mejora interes",
    "28": "Aula invertida (flip-classroom)",
    "29": "GPA ultimo semestre",
    "30": "GPA esperado al graduarse",
}

# All features are ordinal integers — treat as numeric (scaled)
_NUMERIC_COLS = _FEATURE_COLS


def _load() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(_PATH)
    y = (df[_TARGET] >= 1).astype(int)   # 0 = Fail, 1+ = any passing grade
    X = df[_FEATURE_COLS].copy()
    return X, y


def _slider(col: str, lo: int, hi: int, default: int) -> dict:
    return {
        "col": col, "label": _COL_LABELS[col], "type": "slider",
        "min": lo, "max": hi, "default": default,
    }


_UI_FORM = [
    _slider("1",  1, 3, 1),   # age group
    _slider("2",  1, 2, 1),   # sex
    _slider("3",  1, 3, 2),   # hs type
    _slider("4",  1, 5, 1),   # scholarship
    _slider("5",  1, 2, 2),   # additional work
    _slider("6",  1, 2, 2),   # sports/arts
    _slider("7",  1, 2, 2),   # partner
    _slider("8",  1, 5, 1),   # salary
    _slider("9",  1, 4, 1),   # transport
    _slider("10", 1, 4, 2),   # accommodation
    _slider("11", 1, 6, 3),   # mother edu
    _slider("12", 1, 6, 3),   # father edu
    _slider("13", 1, 5, 2),   # siblings
    _slider("14", 1, 3, 1),   # parental status
    _slider("15", 1, 6, 6),   # mother occ
    _slider("16", 1, 5, 5),   # father occ
    _slider("17", 1, 5, 2),   # study hours
    _slider("18", 1, 3, 2),   # reading non-sci
    _slider("19", 1, 3, 2),   # reading sci
    _slider("20", 1, 2, 2),   # seminar attendance
    _slider("21", 1, 3, 1),   # project impact
    _slider("22", 1, 3, 1),   # class attendance
    _slider("23", 1, 3, 1),   # midterm prep mode
    _slider("24", 1, 3, 2),   # midterm prep timing
    _slider("25", 1, 3, 3),   # note taking
    _slider("26", 1, 3, 3),   # listening
    _slider("27", 1, 3, 3),   # discussion
    _slider("28", 1, 3, 3),   # flip classroom
    _slider("29", 1, 5, 3),   # current GPA
    _slider("30", 1, 5, 4),   # expected GPA
]


config = DatasetConfig(
    name          = "Evaluacion Rendimiento Educacion Superior",
    description   = "Predice si un estudiante universitario aprueba el semestre. Chipre, 2019.",
    load_fn       = _load,
    nominal_cols  = [],
    numeric_cols  = _NUMERIC_COLS,
    target_labels = ("Reprueba (GRADE=0)", "Aprueba (GRADE>=1)"),
    ui_form       = _UI_FORM,
)
