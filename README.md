<div align="center">

# Student Academic Performance Predictor

**A multi-dataset, multi-model classification platform for predicting student performance and dropout risk in educational settings.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

This application provides an interactive environment to train, evaluate, and compare supervised classification models across five educational datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). The platform supports one-click model switching, automatic class imbalance correction via SMOTE, stratified cross-validation, and individual student prediction — all through a clean Streamlit interface.

**Key capabilities:**

- Five curated educational datasets, each with a well-defined binary classification target
- Seven classification algorithms available out of the box
- Preprocessing pipeline fully driven by dataset configuration — no hardcoded logic
- SMOTE applied *inside* the cross-validation pipeline (no data leakage)
- `k_neighbors` and fold count computed dynamically to handle small or heavily imbalanced datasets
- Model feature importance visualization adapted per model type (coefficients, Gini, or graceful fallback)
- Individual prediction form auto-generated from dataset metadata

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/your-username/student-dropout-prediction.git
cd student-dropout-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place datasets in data/  (see Datasets section below)

# 4. Launch the app
streamlit run app.py
```

> **Requires Python 3.8+**

---

## Datasets

| # | File | Source | Instances | Features | Target (Class 1) |
|---|------|--------|----------:|:--------:|-----------------|
| 1 | `StudentsPerformance.csv` | [UCI #320](https://archive.ics.uci.edu/dataset/320) — Cortez & Silva, 2008 | 649 | 30 | Final grade G3 ≥ 10 (Pass) |
| 2 | `dropout.csv` | [UCI — SATDAP, 2022](https://archive.ics.uci.edu/dataset/697) | 4,424 | 36 | Student Dropout |
| 3 | `higher_ed.csv` | [UCI #856](https://archive.ics.uci.edu/dataset/856) — Yilmaz & Sekeroglu, 2019 | 145 | 30 | GRADE ≥ 1 (Pass) |
| 4 | `mathe.csv` | [UCI #1031](https://archive.ics.uci.edu/dataset/1031) — Azevedo et al., 2024 | 9,546 | 5 | Correct answer |
| 5 | `academics.arff` | [UCI #467](https://archive.ics.uci.edu/dataset/467) — Hussain et al., 2018 | 131 | 21 | Performance ∈ {Good, Vg, Best} |

**Important:** Datasets 2–5 are included in the `data/` directory.
Dataset 1 (`StudentsPerformance.csv`, semicolon-separated) must be downloaded separately from [UCI #320](https://archive.ics.uci.edu/dataset/320/student+performance) and placed in `data/`.

### Target encoding rationale

| Dataset | Class 0 | Class 1 | Notes |
|---------|---------|---------|-------|
| UCI Student Performance | G3 < 10 | G3 ≥ 10 | G1/G2 excluded (target leakage) |
| Dropout | Graduate / Enrolled | Dropout | `Enrolled` = still active, not yet dropout |
| Higher Ed | Fail (GRADE=0) | Pass (GRADE≥1) | 8/145 samples are Fail — SMOTE required |
| MathE | Incorrect (0) | Correct (1) | Target is natively binary |
| Academics | Pass / Fail | Good / Vg / Best | `esp` column = end-semester performance |

---

## Models

| Model | Interpretability | Feature Importance | Notes |
|-------|:----------------:|:-----------------:|-------|
| Logistic Regression | ★★★★★ | Signed coefficients | Linear baseline; fastest to train |
| CART (Decision Tree) | ★★★★☆ | Gini importance | `max_depth=6` to prevent overfitting |
| Random Forest | ★★★☆☆ | Gini importance | 200 estimators; robust ensemble |
| Gradient Boosting | ★★★☆☆ | Gini importance | 200 estimators, `lr=0.1`, `depth=4` |
| k-Nearest Neighbors | ★★☆☆☆ | — | `k=7`, Minkowski metric |
| Neural Network (MLP) | ★☆☆☆☆ | — | Architecture: `[64 → 32]`, 500 epochs |
| SVM | ★★☆☆☆ | — | RBF kernel, `probability=True` |

---

## Architecture

The project follows **separation of concerns** with four distinct layers. Each layer depends only on abstractions, never on concrete implementations (Dependency Inversion Principle). Adding a new dataset or model requires touching exactly one file.

```
student-dropout-prediction/
│
├── app.py                          # Entry point — sidebar, cache, tab routing only
├── requirements.txt
├── README.md
│
├── data/                           # Raw dataset files (CSV, ARFF)
│
└── src/
    ├── config.py                   # Global constants: RANDOM_STATE, TEST_SIZE, CV_FOLDS
    │
    ├── datasets/                   # DATA LAYER
    │   ├── base.py                 # DatasetConfig — the contract every dataset must fulfill
    │   ├── __init__.py             # DATASET_REGISTRY — unified dataset catalogue
    │   ├── ds1_uci_student.py      # UCI Student Performance
    │   ├── ds2_dropout.py          # Student Dropout Prediction
    │   ├── ds3_higher_ed.py        # Higher Education Performance (Cyprus)
    │   ├── ds4_mathe.py            # MathE Mathematics Platform
    │   └── ds5_academics.py        # Student Academics Performance (ARFF)
    │
    ├── models/                     # MODEL LAYER
    │   └── __init__.py             # MODEL_REGISTRY — all classifiers and their metadata
    │
    ├── core/                       # BUSINESS LOGIC LAYER
    │   ├── pipeline.py             # run_training() — orchestrates the full ML pipeline
    │   ├── preprocessing.py        # build_preprocessor() — OHE + StandardScaler via ColumnTransformer
    │   ├── metrics.py              # confusion matrix, ROC, feature importance, predict_one()
    │   └── __init__.py             # Public API of the core layer
    │
    └── ui/                         # PRESENTATION LAYER
        ├── components.py           # Reusable chart and form rendering functions
        └── pages/
            ├── overview.py         # Tab 1 — Data preview, class distribution, cross-validation
            ├── evaluation.py       # Tab 2 — Confusion matrix, classification report, ROC curve
            ├── features.py         # Tab 3 — Feature importance (model-adaptive)
            └── predict.py          # Tab 4 — Individual prediction form (auto-generated)
```

### Design principles applied

| Principle | Implementation |
|-----------|---------------|
| **Single Responsibility** | `pipeline.py` trains only. `metrics.py` computes only. `components.py` renders only. |
| **Open / Closed** | New dataset = one new file + one registry entry. Zero changes elsewhere. |
| **Dependency Inversion** | `pipeline.py` depends on `DatasetConfig` and `ModelConfig` (abstractions), never on `ds3_higher_ed` or `LogisticRegression` directly. |
| **DRY** | All chart code lives in `components.py`. Prediction forms are auto-generated from declarative `ui_form` metadata — no per-dataset form code. |
| **Fail Fast** | `DatasetConfig` enforces its contract at registration time, not at prediction time. |

---

## Extending the Project

### Adding a new dataset

1. Create `src/datasets/ds6_my_dataset.py` using any existing `dsN_*.py` as a template.
2. Implement `_load() -> Tuple[pd.DataFrame, pd.Series]` returning `(X_features, y_binary)`.
3. Define a `DatasetConfig` with `nominal_cols`, `numeric_cols`, `target_labels`, and `ui_form`.
4. Register it in `src/datasets/__init__.py`:

```python
from src.datasets import ds6_my_dataset

DATASET_REGISTRY[ds6_my_dataset.config.name] = ds6_my_dataset.config
```

That's it. The pipeline, preprocessing, UI form, and prediction logic all adapt automatically.

### Adding a new model

Add one entry to `MODEL_REGISTRY` in `src/models/__init__.py`:

```python
"My Model": ModelConfig(
    name          = "My Model",
    estimator_fn  = lambda: MyClassifier(param=value),
    supports_coef = False,   # True only for linear models with .coef_
    supports_fi   = False,   # True only for tree-based models with .feature_importances_
    description   = "One-line description shown in the UI.",
),
```

---

## Technical notes

### SMOTE and cross-validation

SMOTE is placed **inside** the `imblearn.Pipeline`, ensuring synthetic samples are generated only from training folds and never leak into validation folds. Additionally, `k_neighbors` is computed dynamically:

```
minority_per_fold ≈ minority_train × (n_folds − 1) / n_folds
k_neighbors = max(1, min(5, minority_per_fold − 1))
```

This prevents the `n_neighbors > n_samples` error that occurs with small or severely imbalanced datasets (e.g., Dataset 3 with only 8 minority-class samples).

### Preprocessing strategy

| Feature type | Transformer | Rationale |
|---|---|---|
| Nominal categorical | `OneHotEncoder` | Avoids false ordinal assumption |
| Numeric / ordinal | `StandardScaler` | Required for kNN, SVM, MLP; harmless for trees |
| Binary yes/no strings | Mapped to `{0, 1}` before pipeline | Treated as numeric after mapping |

---

## References

1. Cortez, P. & Silva, A. (2008). *Using data mining to predict secondary school student performance.* FUBUTEC.
2. Realinho, V. et al. (2022). *Predicting Student Dropout and Academic Success.* Data, 7(11), 146.
3. Yilmaz, N. & Sekeroglu, B. (2020). *Student Performance Classification Using Artificial Intelligence Techniques.* ICSCCW 2019, AISC vol. 1095.
4. Flamia Azevedo, B. et al. (2024). *Dataset of mathematics learning and assessment of higher education students using the MathE platform.* Data in Brief.
5. Hussain, S. et al. (2018). *Educational Data Mining and Analysis of Students' Academic Performance Using WEKA.* IJEECS.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The datasets are licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
