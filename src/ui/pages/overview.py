# ─────────────────────────────────────────────────────────────────────────────
#  src/ui/pages/overview.py
#  Tab: Resumen — data preview, class distribution, cross-validation
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st

from src.core.pipeline import TrainingResult
from src.ui.components import render_class_distribution, render_cv_chart


def render(result: TrainingResult) -> None:
    st.header("Resumen del Dataset y Entrenamiento")

    # ── Dataset description ───────────────────────────────────────────────────
    st.markdown(
        "> **{}**  \n{}".format(
            result.dataset_cfg.name,
            result.dataset_cfg.description,
        )
    )
    st.markdown(
        "> **Modelo:** {}  \n{}".format(
            result.model_cfg.name,
            result.model_cfg.description,
        )
    )

    st.divider()

    # ── Data preview ─────────────────────────────────────────────────────────
    st.subheader("Vista previa de los datos")
    X, y = result.dataset_cfg.load()
    preview = X.head(10).copy()
    preview.insert(0, "Target (y)", y.head(10).values)
    st.dataframe(preview, use_container_width=True)
    st.caption(
        "Dataset: {} filas x {} columnas de features.".format(
            X.shape[0], X.shape[1]
        )
    )

    st.divider()

    col_dist, col_cv = st.columns(2, gap="large")

    with col_dist:
        st.subheader("Distribucion de clases")
        render_class_distribution(result)

    with col_cv:
        st.subheader("Validacion Cruzada ({}-Fold)".format(len(result.cv_scores)))
        st.info(
            "SMOTE se aplica **dentro** de cada fold — nunca contamina "
            "el conjunto de validacion.",
            icon="ℹ️",
        )
        render_cv_chart(result)
