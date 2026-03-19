# ─────────────────────────────────────────────────────────────────────────────
#  src/core/metrics.py
#
#  Pure metric functions.  Each takes a TrainingResult and returns a value
#  ready to be consumed by the UI.  No Streamlit imports here.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

from src.core.pipeline import TrainingResult
from src.models import MODEL_REGISTRY


def get_confusion_matrix(result: TrainingResult) -> np.ndarray:
    return confusion_matrix(result.y_test, result.y_pred)


def get_classification_report(result: TrainingResult) -> pd.DataFrame:
    label0, label1 = result.dataset_cfg.target_labels
    report = classification_report(
        result.y_test,
        result.y_pred,
        target_names=[label0, label1],
        output_dict=True,
    )
    return pd.DataFrame(report).transpose()


def get_roc_data(result: TrainingResult) -> Tuple[np.ndarray, np.ndarray, float]:
    """Returns (fpr, tpr, auc_score)."""
    auc = roc_auc_score(result.y_test, result.y_proba)
    fpr, tpr, _ = roc_curve(result.y_test, result.y_proba)
    return fpr, tpr, auc


def get_feature_importance(result: TrainingResult) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [Feature, Score, metric_label].

    Dispatch logic:
      - Logistic Regression  →  signed coefficient  (|coef_| for ranking)
      - Tree-based models    →  feature_importances_ (Gini)
      - kNN / SVM / MLP      →  empty DataFrame (not supported)
    """
    clf   = result.pipeline.named_steps["classifier"]
    names = result.feature_names
    cfg   = result.model_cfg
    n     = len(names)

    if cfg.supports_coef:
        scores = clf.coef_[0][:n]
        metric = "Coeficiente (Reg. Logistica)"
        df = pd.DataFrame({"Feature": names[:len(scores)], "Score": scores})
        df["Abs"] = df["Score"].abs()
        df = df.sort_values("Abs", ascending=False).head(15).drop(columns="Abs")

    elif cfg.supports_fi:
        scores = clf.feature_importances_[:n]
        metric = "Importancia Gini"
        df = pd.DataFrame({"Feature": names[:len(scores)], "Score": scores})
        df = df.sort_values("Score", ascending=False).head(15)

    else:
        return pd.DataFrame(columns=["Feature", "Score", "metric_label"])

    df = df.reset_index(drop=True)
    df["metric_label"] = metric
    return df


def predict_one(result: TrainingResult, raw_input: dict) -> Tuple[int, float]:
    """
    Predict a single observation.

    Parameters
    ----------
    raw_input : dict
        Keys are feature column names; values are the raw (pre-encoding) values
        as collected from the UI form.

    Returns
    -------
    (prediction, probability_of_class_1)
    """
    X_new = pd.DataFrame([raw_input])
    pred  = result.pipeline.predict(X_new)[0]
    proba = result.pipeline.predict_proba(X_new)[0][1]
    return int(pred), float(proba)
