# ─────────────────────────────────────────────────────────────────────────────
#  src/datasets/base.py
#
#  Defines DatasetConfig: the contract that every dataset entry must satisfy.
#  The rest of the system only depends on this interface, never on a specific
#  dataset module — Open/Closed Principle.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple


class DatasetConfig:
    """
    Immutable descriptor for one dataset.

    Parameters
    ----------
    name : str
        Human-readable display name shown in the UI.
    description : str
        One-sentence summary shown in the UI.
    load_fn : Callable[[], Tuple[pd.DataFrame, pd.Series]]
        Zero-argument function that returns (X_raw, y) where:
          - X_raw contains only feature columns (target already removed)
          - y is a binary pd.Series (0 / 1)
    nominal_cols : list of str
        Unordered categorical features -> OneHotEncoder.
    numeric_cols : list of str
        Numeric / already-encoded ordinal features -> StandardScaler.
    target_labels : tuple of (str, str)
        Human-readable names for class 0 and class 1.
    ui_form : list of dicts
        Declarative spec for the prediction form.  Each dict:
          {
            "col":     column name in X_raw,
            "label":   display label,
            "type":    "select" | "slider" | "number",
            # for "select":
            "options": list of raw values,
            "option_labels": list of display labels,
            # for "slider":
            "min": int/float, "max": int/float, "default": int/float,
            # for "number":
            "default": int/float,
          }
    """

    def __init__(
        self,
        name: str,
        description: str,
        load_fn: Callable[[], Tuple[pd.DataFrame, pd.Series]],
        nominal_cols: List[str],
        numeric_cols: List[str],
        target_labels: Tuple[str, str],
        ui_form: List[Dict[str, Any]],
    ):
        self.name          = name
        self.description   = description
        self._load_fn      = load_fn
        self.nominal_cols  = nominal_cols
        self.numeric_cols  = numeric_cols
        self.target_labels = target_labels   # (label_class_0, label_class_1)
        self.ui_form       = ui_form

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Return (X_raw, y).  Delegates to the registered loader."""
        return self._load_fn()

    def __repr__(self) -> str:
        return "DatasetConfig(name={!r})".format(self.name)
