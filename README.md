# Prediccion de Rendimiento Estudiantil

Aplicacion Streamlit con multiples datasets y modelos de clasificacion para
predecir rendimiento y riesgo de abandono en educacion.

---

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Datasets incluidos

| Archivo en `data/`        | Dataset                                    | Target                          |
|---------------------------|--------------------------------------------|---------------------------------|
| `StudentsPerformance.csv` | UCI Student Performance (Cortez, 2008)     | G3 >= 10 → Aprueba              |
| `dropout.csv`             | Student Dropout (UCI, 2022)                | Target == "Dropout"             |
| `higher_ed.csv`           | Higher Ed Performance (Yilmaz, 2019)       | GRADE >= 1 → Aprueba            |
| `mathe.csv`               | MathE Platform (Azevedo, 2024)             | Type of Answer == 1 → Correcto  |
| `academics.arff`          | Student Academics (Hussain, 2018)          | esp in {Good,Vg,Best}           |

> **Nota:** `StudentsPerformance.csv` debe copiarse manualmente a `data/`
> (separador `;`). Los otros 4 archivos ya estan incluidos.

## Agregar un nuevo dataset

1. Crea `src/datasets/ds6_mi_dataset.py` siguiendo la estructura de cualquier
   archivo `dsN_*.py` existente.
2. Define una funcion `_load()` que retorne `(X: DataFrame, y: Series binaria)`.
3. Completa el `DatasetConfig` con `nominal_cols`, `numeric_cols` y `ui_form`.
4. Agrega una linea en `src/datasets/__init__.py`:
   ```python
   from src.datasets import ds6_mi_dataset
   DATASET_REGISTRY[ds6_mi_dataset.config.name] = ds6_mi_dataset.config
   ```
   Nada mas cambia en el proyecto.

## Agregar un nuevo modelo

Agrega una entrada en `src/models/__init__.py`:
```python
"Mi Modelo": ModelConfig(
    name         = "Mi Modelo",
    estimator_fn = lambda: MiClasificador(param=valor),
    supports_coef= False,
    supports_fi  = False,
    description  = "Descripcion breve.",
),
```

## Estructura del proyecto

```
app.py                        # Orquestador: sidebar + cache + 4 tabs
requirements.txt
data/                         # CSVs y ARFFs
src/
  config.py                   # Constantes globales (RANDOM_STATE, etc.)
  datasets/
    base.py                   # DatasetConfig (contrato de datos)
    __init__.py               # DATASET_REGISTRY
    ds1_uci_student.py        # UCI Student Performance
    ds2_dropout.py            # Student Dropout
    ds3_higher_ed.py          # Higher Education Performance
    ds4_mathe.py              # MathE Mathematics
    ds5_academics.py          # Student Academics (ARFF)
  models/
    __init__.py               # MODEL_REGISTRY con ModelConfig
  core/
    pipeline.py               # run_training() + TrainingResult
    preprocessing.py          # ColumnTransformer genererico
    metrics.py                # confusion matrix, ROC, importancia, predict
    __init__.py               # API publica del layer
  ui/
    components.py             # Widgets reutilizables (charts, forms)
    pages/
      overview.py             # Tab 1: Resumen + distribucion + CV
      evaluation.py           # Tab 2: Confusion matrix + reporte + ROC
      features.py             # Tab 3: Importancia de variables
      predict.py              # Tab 4: Prediccion individual
```
