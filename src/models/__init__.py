# ─────────────────────────────────────────────────────────────────────────────
#  src/models/__init__.py
#
#  Model registry.  To add a new model: create a ModelConfig entry here.
#  The pipeline layer never imports sklearn directly — it asks this registry.
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.config import RANDOM_STATE


class ModelConfig:
    """
    Descriptor for one classifier.

    Attributes
    ----------
    name         : display name
    estimator_fn : zero-arg callable returning a fresh sklearn estimator
    supports_coef: True  → logistic regression; show signed coefficients
    supports_fi  : True  → tree-based; show feature_importances_
    description  : one-line model description shown in the UI
    """

    def __init__(self, name, estimator_fn, supports_coef,
                 supports_fi, description):
        self.name          = name
        self._estimator_fn = estimator_fn
        self.supports_coef = supports_coef
        self.supports_fi   = supports_fi
        self.description   = description

    def build_estimator(self):
        """Return a fresh, unfitted estimator instance."""
        return self._estimator_fn()

    def __repr__(self):
        return "ModelConfig(name={!r})".format(self.name)


MODEL_REGISTRY = {
    "Regresion Logistica": ModelConfig(
        name          = "Regresion Logistica",
        estimator_fn  = lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        supports_coef = True,
        supports_fi   = False,
        description   = "Modelo lineal interpretable. Ideal como linea base.",
    ),
    "CART (Arbol de Decision)": ModelConfig(
        name          = "CART (Arbol de Decision)",
        estimator_fn  = lambda: DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        supports_coef = False,
        supports_fi   = True,
        description   = "Arbol de decision con profundidad maxima 6. Muy interpretable.",
    ),
    "Random Forest": ModelConfig(
        name          = "Random Forest",
        estimator_fn  = lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        supports_coef = False,
        supports_fi   = True,
        description   = "Ensemble de 200 arboles. Robusto ante overfitting.",
    ),
    "Gradient Boosting": ModelConfig(
        name          = "Gradient Boosting",
        estimator_fn  = lambda: GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1,
            max_depth=4, random_state=RANDOM_STATE,
        ),
        supports_coef = False,
        supports_fi   = True,
        description   = "Boosting clasico. Alta precision, mas lento de entrenar.",
    ),
    "k-Nearest Neighbors": ModelConfig(
        name          = "k-Nearest Neighbors",
        estimator_fn  = lambda: KNeighborsClassifier(n_neighbors=7, metric="minkowski"),
        supports_coef = False,
        supports_fi   = False,
        description   = "Clasificacion por vecindad. Sensible a la escala de features.",
    ),
    "Red Neuronal (MLP)": ModelConfig(
        name          = "Red Neuronal (MLP)",
        estimator_fn  = lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE,
        ),
        supports_coef = False,
        supports_fi   = False,
        description   = "Perceptron multicapa con capas [64, 32]. Captura relaciones no lineales.",
    ),
    "SVM": ModelConfig(
        name          = "SVM",
        estimator_fn  = lambda: SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        supports_coef = False,
        supports_fi   = False,
        description   = "Maquina de soporte vectorial con kernel RBF.",
    ),
}

MODEL_NAMES = list(MODEL_REGISTRY.keys())
