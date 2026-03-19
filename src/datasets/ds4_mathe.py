# ─────────────────────────────────────────────────────────────────────────────
#  src/datasets/ds4_mathe.py
#  Assessing Mathematics Learning in Higher Education — MathE Platform, 2024
#  Target: Type of Answer — 1=correct, 0=incorrect
#
#  Student ID and Question ID are identifiers — excluded.
#  Question Level, Topic, Subtopic, Keywords are nominal.
#  Student Country is nominal.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
from typing import Tuple

from src.datasets.base import DatasetConfig

_PATH      = "data/mathe.csv"
_TARGET    = "Type of Answer"
_EXCLUDED  = ["Student ID", "Question ID"]

_NOMINAL_COLS = [
    "Student Country",
    "Question Level",
    "Topic",
    "Subtopic",
    "Keywords",
]
_NUMERIC_COLS: list = []    # no numeric features after removing IDs


def _load() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(_PATH, sep=";", encoding="latin-1")
    y  = df[_TARGET].astype(int)
    X  = df[_NOMINAL_COLS].copy()
    return X, y


_UI_FORM = [
    {
        "col": "Student Country", "label": "Pais del estudiante",
        "type": "select",
        "options":       ["Portugal", "Ireland", "Italy", "Spain", "Other"],
        "option_labels": ["Portugal", "Irlanda",  "Italia","Espana", "Otro"],
    },
    {
        "col": "Question Level", "label": "Nivel de la pregunta",
        "type": "select",
        "options":       ["Basic", "Advanced"],
        "option_labels": ["Basico", "Avanzado"],
    },
    {
        "col": "Topic", "label": "Tema matematico",
        "type": "select",
        "options":       ["Statistics", "Calculus", "Algebra", "Geometry", "Other"],
        "option_labels": ["Estadistica", "Calculo", "Algebra", "Geometria", "Otro"],
    },
    {
        "col": "Subtopic", "label": "Subtema",
        "type": "select",
        "options":       ["Statistics", "Calculus", "Linear Algebra", "Other"],
        "option_labels": ["Estadistica", "Calculo",  "Algebra Lineal", "Otro"],
    },
    {
        "col": "Keywords", "label": "Palabras clave",
        "type": "select",
        "options":       [
            "Stem and Leaf diagram,Relative frequency,Sample,Frequency",
            "Derivative,Function,Limit",
            "Matrix,Vector,Eigenvalue",
            "Other",
        ],
        "option_labels": [
            "Diagrama hojas / frecuencia",
            "Derivada / funcion / limite",
            "Matriz / vector / valor propio",
            "Otro",
        ],
    },
]


config = DatasetConfig(
    name          = "Aprendizaje Matematicas (MathE)",
    description   = "Predice si un estudiante responde correctamente una pregunta matematica.",
    load_fn       = _load,
    nominal_cols  = _NOMINAL_COLS,
    numeric_cols  = _NUMERIC_COLS,
    target_labels = ("Respuesta incorrecta", "Respuesta correcta"),
    ui_form       = _UI_FORM,
)
