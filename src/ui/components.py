# ─────────────────────────────────────────────────────────────────────────────
#  src/ui/components.py
#
#  Reusable Streamlit rendering functions.
#  All chart-drawing and widget helpers live here so that page modules stay
#  thin: they only call these functions and handle user interaction.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

from src.core.pipeline import TrainingResult

# ── Shared palette ────────────────────────────────────────────────────────────

C = {
    "primary": "#1a56db",
    "success": "#057a55",
    "danger":  "#c81e1e",
    "neutral": "#6b7280",
}


# ── KPI strip ─────────────────────────────────────────────────────────────────

def render_kpi_strip(result: TrainingResult) -> None:
    """Four-column KPI bar always visible at the top of every page."""
    cv = result.cv_scores
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modelo",              result.model_cfg.name)
    c2.metric("Dataset",             result.dataset_cfg.name)
    c3.metric("AUC-ROC CV (media)",  "{:.3f}".format(cv.mean()))
    c4.metric("AUC-ROC CV (std)",    "+/- {:.3f}".format(cv.std()))


# ── Distribution bar chart ────────────────────────────────────────────────────

def render_class_distribution(result: TrainingResult) -> None:
    counts = result.class_counts
    label0, label1 = result.dataset_cfg.target_labels
    ratio  = result.imbalance_ratio

    c1, c2 = st.columns(2)
    c1.metric(label0, counts.get(0, 0))
    c2.metric(label1, counts.get(1, 0))

    if ratio < 0.6:
        st.warning(
            "Desbalance detectado (ratio = {:.2f}). "
            "SMOTE corrige esto dentro del pipeline.".format(ratio)
        )
    else:
        st.success("Clases balanceadas (ratio = {:.2f}).".format(ratio))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(
        [label0, label1],
        [counts.get(0, 0), counts.get(1, 0)],
        color=[C["danger"], C["success"]],
        edgecolor="white", linewidth=0.8,
    )
    ax.set_ylabel("Instancias")
    ax.set_title("Distribucion de clases")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    st.pyplot(fig)
    plt.close(fig)


# ── Cross-validation bar chart ────────────────────────────────────────────────

def render_cv_chart(result: TrainingResult) -> None:
    cv    = result.cv_scores
    folds = ["Fold {}".format(i + 1) for i in range(len(cv))]

    cv_df = pd.DataFrame({"Fold": folds, "AUC-ROC": cv})
    st.dataframe(cv_df.set_index("Fold"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(folds, cv, color=C["primary"], alpha=0.85)
    ax.axhline(
        cv.mean(), color=C["danger"], linestyle="--",
        label="Media = {:.3f}".format(cv.mean()),
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC por fold — {}".format(result.model_cfg.name))
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


# ── Confusion matrix heatmap ──────────────────────────────────────────────────

def render_confusion_matrix(cm: np.ndarray, result: TrainingResult) -> None:
    label0, label1 = result.dataset_cfg.target_labels
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=[label0, label1],
        yticklabels=[label0, label1],
    )
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Valor real")
    ax.set_title("Matriz de Confusion — {}".format(result.model_cfg.name))
    st.pyplot(fig)
    plt.close(fig)


# ── ROC curve ─────────────────────────────────────────────────────────────────

def render_roc_curve(fpr, tpr, auc: float, result: TrainingResult) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=C["primary"], lw=2,
            label="AUC = {:.3f}".format(auc))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatorio")
    ax.fill_between(fpr, tpr, alpha=0.08, color=C["primary"])
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title("Curva ROC — {}".format(result.model_cfg.name))
    ax.legend(loc="lower right")
    st.pyplot(fig)
    plt.close(fig)


# ── Feature importance bar chart ──────────────────────────────────────────────

def render_feature_importance(imp_df: pd.DataFrame,
                               result: TrainingResult) -> None:
    if imp_df.empty:
        st.info(
            "**{}** no expone importancia de variables directamente.  \n"
            "Prueba con Regresion Logistica, CART, Random Forest "
            "o Gradient Boosting.".format(result.model_cfg.name)
        )
        return

    metric_label = imp_df["metric_label"].iloc[0]
    st.caption(metric_label)

    # For LR use signed values; for tree-based always positive
    values = imp_df["Score"].values
    if result.model_cfg.supports_coef:
        colors = [C["success"] if v > 0 else C["danger"] for v in values]
    else:
        colors = C["primary"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp_df["Feature"], values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(metric_label)
    ax.set_title("Top variables — {}".format(result.model_cfg.name))
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close(fig)

    display_cols = ["Feature", "Score"]
    st.dataframe(
        imp_df[display_cols].rename(columns={"Score": metric_label}),
        use_container_width=True,
    )


# ── Dynamic prediction form ───────────────────────────────────────────────────

def render_prediction_form(result: TrainingResult) -> dict:
    """
    Renders the prediction form driven by dataset_cfg.ui_form.
    Returns a dict {col: raw_value} ready to pass to predict_one().
    """
    ui_form = result.dataset_cfg.ui_form
    n_cols  = 3
    cols    = st.columns(n_cols)
    values  = {}

    for idx, field in enumerate(ui_form):
        col     = field["col"]
        label   = field["label"]
        ftype   = field["type"]
        target  = cols[idx % n_cols]

        with target:
            if ftype == "select":
                options       = field["options"]
                option_labels = field["option_labels"]
                label_to_raw  = dict(zip(option_labels, options))
                chosen_label  = st.selectbox(
                    label, option_labels, key="pred_{}".format(col)
                )
                values[col] = label_to_raw[chosen_label]

            elif ftype == "slider":
                values[col] = st.slider(
                    label,
                    min_value=field["min"],
                    max_value=field["max"],
                    value=field["default"],
                    key="pred_{}".format(col),
                )
            else:  # "number"
                values[col] = st.number_input(
                    label, value=field.get("default", 0),
                    key="pred_{}".format(col),
                )

    return values
