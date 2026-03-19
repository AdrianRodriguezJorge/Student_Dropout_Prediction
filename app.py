# ─────────────────────────────────────────────────────────────────────────────
#  app.py — Entry point.  Thin orchestrator: selectors + cache + tab routing.
#           No business logic lives here.
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st

from src.core.pipeline import run_training
from src.datasets import DATASET_REGISTRY, DATASET_NAMES
from src.models import MODEL_REGISTRY, MODEL_NAMES
from src.ui.components import render_kpi_strip
from src.ui.pages import overview, evaluation, features, predict

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Rendimiento Estudiantil",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar: selectors ────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Configuracion")

    selected_dataset = st.selectbox(
        "Dataset",
        DATASET_NAMES,
        help="Elige el conjunto de datos a analizar.",
    )
    selected_model = st.selectbox(
        "Modelo",
        MODEL_NAMES,
        help="El modelo se entrena con SMOTE + CV estratificada.",
    )

    dataset_cfg = DATASET_REGISTRY[selected_dataset]
    model_cfg   = MODEL_REGISTRY[selected_model]

    st.divider()
    st.caption(dataset_cfg.description)
    st.caption("**Modelo:** " + model_cfg.description)

# ── Training (cached per dataset+model pair) ──────────────────────────────────

@st.cache_resource(show_spinner="Entrenando modelo...")
def get_result(dataset_name: str, model_name: str):
    return run_training(
        DATASET_REGISTRY[dataset_name],
        MODEL_REGISTRY[model_name],
    )


result = get_result(selected_dataset, selected_model)

# ── Header + KPI strip ────────────────────────────────────────────────────────

st.title("Prediccion de Rendimiento Estudiantil")
render_kpi_strip(result)
st.divider()

# ── Tab navigation (4 tabs replace 9 sidebar pages) ──────────────────────────

tab_overview, tab_eval, tab_feat, tab_pred = st.tabs([
    "Resumen",
    "Evaluacion del Modelo",
    "Importancia de Variables",
    "Predecir",
])

with tab_overview:
    overview.render(result)

with tab_eval:
    evaluation.render(result)

with tab_feat:
    features.render(result)

with tab_pred:
    predict.render(result)
