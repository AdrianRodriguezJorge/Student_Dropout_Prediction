# ─────────────────────────────────────────────────────────────────────────────
#  src/datasets/ds5_academics.py
#  Student Academics Performance — Hussain et al., 2018  (.arff format)
#  Target: esp (end-semester performance) — Good/Vg/Best → Pass (1),
#                                           Pass/Fail     → 0
#
#  All 21 non-target features are nominal strings → all go through OHE.
# ─────────────────────────────────────────────────────────────────────────────

import io
import pandas as pd
from typing import Tuple

from src.datasets.base import DatasetConfig

_PATH = "data/academics.arff"

_TARGET = "esp"

_FEATURE_COLS = [
    "ge", "cst", "tnp", "twp", "iap",
    "arr", "ms", "ls", "as", "fmi", "fs",
    "fq", "mq", "fo", "mo", "nf", "sh",
    "ss", "me", "tt", "atd",
]

_NOMINAL_COLS = _FEATURE_COLS
_NUMERIC_COLS: list = []


def _parse_arff(path: str) -> pd.DataFrame:
    """Minimal ARFF parser — reads @ATTRIBUTE names then @DATA block."""
    with open(path, "r") as fh:
        lines = fh.readlines()

    attr_names = []
    data_lines = []
    in_data = False

    for line in lines:
        stripped = line.strip()
        upper    = stripped.upper()
        if upper == "@DATA":
            in_data = True
            continue
        if upper.startswith("@ATTRIBUTE"):
            parts = stripped.split()
            attr_names.append(parts[1])
        elif in_data and stripped and not stripped.startswith("%"):
            data_lines.append(stripped)

    rows = [row.split(",") for row in data_lines]
    return pd.DataFrame(rows, columns=attr_names)


def _load() -> Tuple[pd.DataFrame, pd.Series]:
    df = _parse_arff(_PATH)

    # esp: Best/Vg/Good → 1 (pass),  Pass/Fail → 0
    y = df[_TARGET].isin(["Best", "Vg", "Good"]).astype(int)
    X = df[_FEATURE_COLS].copy()
    return X, y


_LABEL_MAPS = {
    "ge":  (["M", "F"],                                       ["Masculino", "Femenino"]),
    "cst": (["G", "ST", "SC", "OBC", "MOBC"],                ["General", "ST", "SC", "OBC", "MOBC"]),
    "tnp": (["Best", "Vg", "Good", "Pass", "Fail"],          ["Excelente", "Muy bueno", "Bueno", "Suficiente", "Reprobado"]),
    "twp": (["Best", "Vg", "Good", "Pass", "Fail"],          ["Excelente", "Muy bueno", "Bueno", "Suficiente", "Reprobado"]),
    "iap": (["Best", "Vg", "Good", "Pass", "Fail"],          ["Excelente", "Muy bueno", "Bueno", "Suficiente", "Reprobado"]),
    "arr": (["Y", "N"],                                       ["Si", "No"]),
    "ms":  (["Married", "Unmarried"],                         ["Casado", "Soltero"]),
    "ls":  (["T", "V"],                                       ["Ciudad", "Pueblo"]),
    "as":  (["Free", "Paid"],                                 ["Gratuito", "Pagado"]),
    "fmi": (["Vh", "High", "Am", "Medium", "Low"],           ["Muy alto", "Alto", "Medio", "Moderado", "Bajo"]),
    "fs":  (["Large", "Average", "Small"],                    ["Grande", "Promedio", "Pequeno"]),
    "fq":  (["Il", "Um", "10", "12", "Degree", "Pg"],        ["Analfabeto", "Sin calificar", "10", "12", "Grado", "Posgrado"]),
    "mq":  (["Il", "Um", "10", "12", "Degree", "Pg"],        ["Analfabeto", "Sin calificar", "10", "12", "Grado", "Posgrado"]),
    "fo":  (["Service", "Business", "Retired", "Farmer", "Others"], ["Servicio", "Negocio", "Jubilado", "Agricultor", "Otro"]),
    "mo":  (["Service", "Business", "Retired", "Housewife", "Others"], ["Servicio", "Negocio", "Jubilada", "Ama de casa", "Otro"]),
    "nf":  (["Large", "Average", "Small"],                    ["Grande", "Promedio", "Pequena"]),
    "sh":  (["Good", "Average", "Poor"],                      ["Bueno", "Promedio", "Pobre"]),
    "ss":  (["Govt", "Private"],                              ["Gobierno", "Privado"]),
    "me":  (["Eng", "Asm", "Hin", "Ben"],                    ["Ingles", "Asamesa", "Hindi", "Bengali"]),
    "tt":  (["Large", "Average", "Small"],                    ["Grande", "Promedio", "Pequeno"]),
    "atd": (["Good", "Average", "Poor"],                      ["Buena", "Promedio", "Pobre"]),
}

_COL_DISPLAY = {
    "ge":  "Genero",           "cst": "Casta",
    "tnp": "Nota teorica prev","twp": "Nota practica prev",
    "iap": "Nota practicas int","arr": "Tiene atrasos",
    "ms":  "Estado civil",     "ls":  "Ubicacion",
    "as":  "Admision",         "fmi": "Ingresos familiares",
    "fs":  "Tamano familia",   "fq":  "Educacion padre",
    "mq":  "Educacion madre",  "fo":  "Ocupacion padre",
    "mo":  "Ocupacion madre",  "nf":  "Numero familia",
    "sh":  "Situacion hogar",  "ss":  "Tipo escuela",
    "me":  "Idioma materno",   "tt":  "Tamano ciudad",
    "atd": "Asistencia",
}

_UI_FORM = [
    {
        "col": col,
        "label": _COL_DISPLAY[col],
        "type": "select",
        "options":       _LABEL_MAPS[col][0],
        "option_labels": _LABEL_MAPS[col][1],
    }
    for col in _FEATURE_COLS
]


config = DatasetConfig(
    name          = "Rendimiento Academico Estudiantil",
    description   = "Predice el rendimiento final de estudiantes (India). Hussain et al., 2018.",
    load_fn       = _load,
    nominal_cols  = _NOMINAL_COLS,
    numeric_cols  = _NUMERIC_COLS,
    target_labels = ("Rendimiento bajo (Pass/Fail)", "Rendimiento alto (Good/Vg/Best)"),
    ui_form       = _UI_FORM,
)
