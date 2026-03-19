# ─────────────────────────────────────────────────────────────────────────────
#  src/ui/pages/features.py
#  Tab: Variables — feature importance
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st

from src.core import get_feature_importance
from src.core.pipeline import TrainingResult
from src.ui.components import render_feature_importance


def render(result: TrainingResult) -> None:
    st.header("Importancia de Variables")
    st.markdown(
        "Muestra qué features tienen mayor influencia en la prediccion.  \n"
        "- **Regresion Logistica**: coeficientes firmados "
        "(verde = aumenta probabilidad clase 1, rojo = la reduce).  \n"
        "- **Arboles / Bosques / GB**: importancia Gini (siempre positiva).  \n"
        "- **kNN / SVM / MLP**: no disponible para este tipo de modelo."
    )

    imp_df = get_feature_importance(result)
    render_feature_importance(imp_df, result)
