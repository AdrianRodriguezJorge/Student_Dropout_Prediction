# ─────────────────────────────────────────────────────────────────────────────
#  src/ui/pages/evaluation.py
#  Tab: Evaluacion — confusion matrix, classification report, ROC curve
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st

from src.core import (
    get_confusion_matrix,
    get_classification_report,
    get_roc_data,
)
from src.core.pipeline import TrainingResult
from src.ui.components import (
    render_confusion_matrix,
    render_roc_curve,
)


def render(result: TrainingResult) -> None:
    st.header("Evaluacion del Modelo")
    st.caption(
        "Todas las metricas se calculan sobre el conjunto de prueba "
        "({}% del dataset, nunca visto durante el entrenamiento).".format(
            int(20)
        )
    )

    # ── Row 1: confusion matrix + classification report ───────────────────────
    col_cm, col_rep = st.columns(2, gap="large")

    with col_cm:
        st.subheader("Matriz de Confusion")
        cm = get_confusion_matrix(result)
        render_confusion_matrix(cm, result)

    with col_rep:
        st.subheader("Reporte de Clasificacion")
        report_df = get_classification_report(result)
        st.dataframe(
            report_df.style.format(precision=3),
            use_container_width=True,
        )
        label0, label1 = result.dataset_cfg.target_labels
        st.caption(
            "Clase 0 = {}  |  Clase 1 = {}".format(label0, label1)
        )

    st.divider()

    # ── Row 2: ROC curve ──────────────────────────────────────────────────────
    st.subheader("Curva ROC")
    fpr, tpr, auc = get_roc_data(result)
    st.metric("AUC-ROC (test set)", "{:.3f}".format(auc))
    render_roc_curve(fpr, tpr, auc, result)
