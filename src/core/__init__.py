# ─────────────────────────────────────────────────────────────────────────────
#  src/core/__init__.py  — Public API of the core layer
# ─────────────────────────────────────────────────────────────────────────────

from src.core.pipeline import run_training, TrainingResult
from src.core.metrics import (
    get_confusion_matrix,
    get_classification_report,
    get_roc_data,
    get_feature_importance,
    predict_one,
)
