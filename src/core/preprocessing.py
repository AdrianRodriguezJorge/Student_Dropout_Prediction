# ─────────────────────────────────────────────────────────────────────────────
#  src/core/preprocessing.py
#
#  Builds a ColumnTransformer from a DatasetConfig.
#  Pure functions — no global state, no imports of specific datasets.
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.datasets.base import DatasetConfig


def build_preprocessor(cfg: DatasetConfig) -> ColumnTransformer:
    """
    Returns an *unfitted* ColumnTransformer driven by cfg.

    Strategy
    --------
    nominal_cols -> OneHotEncoder  (no ordinal assumption)
    numeric_cols -> StandardScaler (benefits kNN, SVM, MLP; harmless for trees)

    If a category list is empty the transformer is simply omitted so that
    sklearn does not raise on an empty selection.
    """
    transformers = []

    if cfg.nominal_cols:
        transformers.append((
            "nominal",
            Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore",
                                            sparse_output=False))]),
            cfg.nominal_cols,
        ))

    if cfg.numeric_cols:
        transformers.append((
            "numeric",
            Pipeline([("scaler", StandardScaler())]),
            cfg.numeric_cols,
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_feature_names(preprocessor: ColumnTransformer,
                      cfg: DatasetConfig):
    """
    Returns the list of feature names after the preprocessor has been fitted.
    Safe to call even if one of the two transformers is absent.
    """
    names = []

    if "nominal" in preprocessor.named_transformers_:
        ohe = preprocessor.named_transformers_["nominal"].named_steps["ohe"]
        names += ohe.get_feature_names_out(cfg.nominal_cols).tolist()

    if "numeric" in preprocessor.named_transformers_:
        names += cfg.numeric_cols

    return names
