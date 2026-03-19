# ─────────────────────────────────────────────────────────────────────────────
#  src/datasets/ds1_uci_student.py
#  UCI Student Performance — Cortez & Silva, 2008
#  Target: G3 >= 10 → Pass (1) / Fail (0)
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
from typing import Tuple

from src.datasets.base import DatasetConfig

_PATH = "data/StudentsPerformance.csv"

# G1/G2 excluded: period grades that would constitute target leakage when the
# goal is to predict risk at enrollment time.
_EXCLUDED = ["G1", "G2"]
_TARGET   = "G3"
_THRESHOLD = 10

_BINARY_COLS = [
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]
_NOMINAL_COLS = [
    "school", "sex", "address", "famsize",
    "Pstatus", "Mjob", "Fjob", "reason", "guardian",
]
_NUMERIC_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences",
]


def _load() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(_PATH, sep=";")
    df.columns = [c.strip() for c in df.columns]

    y = (df[_TARGET] >= _THRESHOLD).astype(int)

    # Map yes/no binary columns to 1/0 before handing off to the pipeline
    X = df.drop(columns=_EXCLUDED + [_TARGET], errors="ignore").copy()
    for col in _BINARY_COLS:
        X[col] = X[col].map({"yes": 1, "no": 0})

    return X, y


def _yn_form(col: str, label: str) -> dict:
    return {
        "col": col, "label": label, "type": "select",
        "options": ["yes", "no"], "option_labels": ["Si", "No"],
    }


_UI_FORM = [
    {"col": "school",   "label": "Escuela",            "type": "select",
     "options": ["GP", "MS"], "option_labels": ["Gabriel Pereira", "Mousinho da Silveira"]},
    {"col": "sex",      "label": "Sexo",                "type": "select",
     "options": ["F", "M"], "option_labels": ["Femenino", "Masculino"]},
    {"col": "age",      "label": "Edad",                "type": "slider",  "min": 15, "max": 22, "default": 17},
    {"col": "address",  "label": "Residencia",          "type": "select",
     "options": ["U", "R"], "option_labels": ["Urbano", "Rural"]},
    {"col": "famsize",  "label": "Tamano familiar",     "type": "select",
     "options": ["LE3", "GT3"], "option_labels": ["3 o menos", "Mas de 3"]},
    {"col": "Pstatus",  "label": "Estado parental",     "type": "select",
     "options": ["T", "A"], "option_labels": ["Juntos", "Separados"]},
    {"col": "Medu",     "label": "Educacion madre (0-4)","type": "slider", "min": 0, "max": 4, "default": 2},
    {"col": "Fedu",     "label": "Educacion padre (0-4)","type": "slider", "min": 0, "max": 4, "default": 2},
    {"col": "Mjob",     "label": "Trabajo madre",       "type": "select",
     "options": ["teacher","health","services","at_home","other"],
     "option_labels": ["Docente","Salud","Servicios","En casa","Otro"]},
    {"col": "Fjob",     "label": "Trabajo padre",       "type": "select",
     "options": ["teacher","health","services","at_home","other"],
     "option_labels": ["Docente","Salud","Servicios","En casa","Otro"]},
    {"col": "reason",   "label": "Razon de eleccion",   "type": "select",
     "options": ["home","reputation","course","other"],
     "option_labels": ["Cercania","Reputacion","Curso","Otro"]},
    {"col": "guardian", "label": "Tutor",               "type": "select",
     "options": ["mother","father","other"],
     "option_labels": ["Madre","Padre","Otro"]},
    {"col": "traveltime","label": "Traslado (1-4)",     "type": "slider",  "min": 1, "max": 4, "default": 1},
    {"col": "studytime", "label": "Estudio semanal (1-4)","type": "slider","min": 1, "max": 4, "default": 2},
    {"col": "failures",  "label": "Reprobados previos", "type": "slider",  "min": 0, "max": 4, "default": 0},
    _yn_form("schoolsup", "Apoyo extra escuela"),
    _yn_form("famsup",    "Apoyo familiar"),
    _yn_form("paid",      "Clases pagadas extra"),
    _yn_form("activities","Actividades extracurriculares"),
    _yn_form("nursery",   "Asistio a guarderia"),
    _yn_form("higher",    "Quiere educacion superior"),
    _yn_form("internet",  "Internet en casa"),
    _yn_form("romantic",  "Relacion romantica"),
    {"col": "famrel",   "label": "Relacion familiar (1-5)", "type": "slider","min": 1,"max": 5,"default": 3},
    {"col": "freetime", "label": "Tiempo libre (1-5)",       "type": "slider","min": 1,"max": 5,"default": 3},
    {"col": "goout",    "label": "Sale con amigos (1-5)",    "type": "slider","min": 1,"max": 5,"default": 3},
    {"col": "Dalc",     "label": "Alcohol laboral (1-5)",    "type": "slider","min": 1,"max": 5,"default": 1},
    {"col": "Walc",     "label": "Alcohol fin semana (1-5)", "type": "slider","min": 1,"max": 5,"default": 1},
    {"col": "health",   "label": "Salud (1-5)",              "type": "slider","min": 1,"max": 5,"default": 3},
    {"col": "absences", "label": "Ausencias",                "type": "slider","min": 0,"max": 93,"default": 5},
]


config = DatasetConfig(
    name          = "UCI Student Performance",
    description   = "Predice si un estudiante aprueba el curso (G3 >= 10). Portugal, 2008.",
    load_fn       = _load,
    nominal_cols  = _NOMINAL_COLS,
    numeric_cols  = _NUMERIC_COLS + _BINARY_COLS,
    target_labels = ("Reprueba (G3 < 10)", "Aprueba (G3 >= 10)"),
    ui_form       = _UI_FORM,
)
