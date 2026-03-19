# ─────────────────────────────────────────────────────────────────────────────
#  src/ui/pages/predict.py
#  Tab: Predecir — individual student prediction form
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st

from src.core import predict_one
from src.core.pipeline import TrainingResult
from src.ui.components import render_prediction_form


def render(result: TrainingResult) -> None:
    st.header("Prediccion Individual")
    label0, label1 = result.dataset_cfg.target_labels
    st.info(
        "Complete los datos del estudiante.  \n"
        "El modelo **{}** predecira a cual clase pertenece:  \n"
        "- Clase 0 → {}  \n"
        "- Clase 1 → {}".format(result.model_cfg.name, label0, label1),
        icon="🎓",
    )

    with st.form("predict_form"):
        raw_values = render_prediction_form(result)
        submitted  = st.form_submit_button(
            "Predecir", use_container_width=True, type="primary"
        )

    if submitted:
        pred, proba = predict_one(result, raw_values)
        label       = label1 if pred == 1 else label0

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Resultado",               label)
        m2.metric("Probabilidad (clase 1)",  "{:.1f}%".format(proba * 100))

        if pred == 1:
            st.success(
                "El modelo predice **{}**.".format(label1), icon="✅"
            )
        else:
            st.error(
                "El modelo predice **{}**. "
                "Se recomienda intervencion preventiva.".format(label0),
                icon="⚠️",
            )
