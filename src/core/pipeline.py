# ─────────────────────────────────────────────────────────────────────────────
#  src/core/pipeline.py
#
#  Orchestrates the full training flow:
#    load data → split → build pipeline → cross-validate → fit → predict
#
#  Depends only on abstractions (DatasetConfig, ModelConfig) — never on a
#  specific dataset or model module.  Single Responsibility: training only.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from typing import List, Tuple

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)

from src.config import RANDOM_STATE, TEST_SIZE, CV_FOLDS
from src.datasets.base import DatasetConfig
from src.models import ModelConfig
from src.core.preprocessing import build_preprocessor, get_feature_names


class TrainingResult:
    """
    Value object that carries every artifact produced during training.
    Immutable after construction — the UI reads from it, never writes to it.
    """

    def __init__(
        self,
        pipeline,           # fitted imblearn.Pipeline
        feature_names,      # List[str]
        X_test,             # pd.DataFrame
        y_test,             # pd.Series
        y_pred,             # np.ndarray
        y_proba,            # np.ndarray  (probability of class 1)
        cv_scores,          # np.ndarray  (AUC per fold)
        class_counts,       # dict  {0: n, 1: m}
        imbalance_ratio,    # float  minority / majority
        dataset_cfg,        # DatasetConfig
        model_cfg,          # ModelConfig
    ):
        self.pipeline        = pipeline
        self.feature_names   = feature_names
        self.X_test          = X_test
        self.y_test          = y_test
        self.y_pred          = y_pred
        self.y_proba         = y_proba
        self.cv_scores       = cv_scores
        self.class_counts    = class_counts
        self.imbalance_ratio = imbalance_ratio
        self.dataset_cfg     = dataset_cfg
        self.model_cfg       = model_cfg


def _class_balance(y: pd.Series) -> Tuple[dict, float]:
    counts   = y.value_counts().to_dict()
    minority = min(counts.values())
    majority = max(counts.values())
    return counts, minority / majority


def run_training(dataset_cfg: DatasetConfig,
                 model_cfg: ModelConfig) -> TrainingResult:
    """
    Full training routine.

    Pipeline order inside imblearn.Pipeline:
        preprocessor  →  SMOTE  →  classifier

    SMOTE is placed *inside* the pipeline so it is applied only on the
    training fold during cross-validation, never on the validation fold.
    This prevents data leakage.
    """

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X, y = dataset_cfg.load()
    class_counts, ratio = _class_balance(y)

    # ── 2. Stratified split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )

    # ── 3. Build pipeline ─────────────────────────────────────────────────────
    preprocessor = build_preprocessor(dataset_cfg)
    estimator    = model_cfg.build_estimator()

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote",        SMOTE(random_state=RANDOM_STATE)),
        ("classifier",   estimator),
    ])

    # ── 4. Cross-validation (AUC-ROC) on training set only ───────────────────
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="roc_auc"
    )

    # ── 5. Final fit on full training set ────────────────────────────────────
    pipeline.fit(X_train, y_train)

    # ── 6. Predictions on held-out test set ──────────────────────────────────
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # ── 7. Recover feature names from fitted preprocessor ────────────────────
    feature_names = get_feature_names(
        pipeline.named_steps["preprocessor"], dataset_cfg
    )

    return TrainingResult(
        pipeline        = pipeline,
        feature_names   = feature_names,
        X_test          = X_test,
        y_test          = y_test,
        y_pred          = y_pred,
        y_proba         = y_proba,
        cv_scores       = cv_scores,
        class_counts    = class_counts,
        imbalance_ratio = ratio,
        dataset_cfg     = dataset_cfg,
        model_cfg       = model_cfg,
    )
