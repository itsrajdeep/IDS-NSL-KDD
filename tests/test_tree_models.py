"""
test_tree_models.py — Unit tests for tree and instance ensemble models.

Verifies:
  1. Auto-registration in MODEL_REGISTRY
  2. Contract inheritance and interface compliance (fit, predict, evaluate)
  3. Binary classification (Level 1 Edge scenario)
  4. Multiclass classification (Level 2 Cloud scenario)
  5. Metrics dictionary structure and positive latency timing
"""

import pytest
import numpy as np

from src.contracts import BaseIDSModel
from src.models.base import IDSModelMixin
from src.models import MODEL_REGISTRY
from src.models.tree_models import (
    DecisionTreeModel,
    RandomForestModel,
    ExtraTreesModel,
    BaselineNBModel,
)


@pytest.fixture
def synthetic_binary_data():
    """Synthetic binary dataset (Level 1 Edge: Normal vs Attack)."""
    np.random.seed(42)
    X_train = np.random.randn(100, 15)
    y_train = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
    X_test = np.random.randn(30, 15)
    y_test = np.random.choice([0, 1], size=30, p=[0.7, 0.3])
    return X_train, y_train, X_test, y_test


@pytest.fixture
def synthetic_multiclass_data():
    """Synthetic 4-class dataset (Level 2 Cloud: DoS, Probe, R2L, U2R)."""
    np.random.seed(42)
    classes = np.array(["DoS", "Probe", "R2L", "U2R"])
    X_train = np.random.randn(120, 15)
    y_train = np.random.choice(classes, size=120, p=[0.5, 0.3, 0.15, 0.05])
    X_test = np.random.randn(40, 15)
    y_test = np.random.choice(classes, size=40, p=[0.5, 0.3, 0.15, 0.05])
    return X_train, y_train, X_test, y_test


# Model classes assigned to 4kub0
ALL_TREE_CLASSES = [
    DecisionTreeModel,
    RandomForestModel,
    ExtraTreesModel,
    BaselineNBModel,
]


def test_model_registry_discovery():
    """All 4 tree models must be automatically registered in MODEL_REGISTRY."""
    expected_names = ["DecisionTree", "RandomForest", "ExtraTrees", "NaiveBayes"]
    for name in expected_names:
        assert name in MODEL_REGISTRY, f"{name} not discovered in MODEL_REGISTRY"
        assert issubclass(MODEL_REGISTRY[name], BaseIDSModel)
        assert issubclass(MODEL_REGISTRY[name], IDSModelMixin)


@pytest.mark.parametrize("model_cls", ALL_TREE_CLASSES)
def test_contract_compliance(model_cls):
    """Verify class attributes and contract conformance."""
    model = model_cls()
    assert isinstance(model.name, str) and len(model.name) > 0
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "evaluate")


@pytest.mark.parametrize("model_cls", ALL_TREE_CLASSES)
def test_binary_fit_predict_evaluate(model_cls, synthetic_binary_data):
    """Test Level 1 binary classification pipeline and metric outputs."""
    X_train, y_train, X_test, y_test = synthetic_binary_data
    model = model_cls()

    # 1. Fit
    model.fit(X_train, y_train)

    # 2. Predict
    preds = model.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X_test),)
    assert set(preds).issubset(set(np.unique(y_train)))

    # 3. Evaluate
    metrics = model.evaluate(X_test, y_test)
    assert isinstance(metrics, dict)
    required_keys = {"accuracy", "precision", "recall", "f1", "fpr", "latency_us"}
    assert required_keys == set(metrics.keys())

    # Check value bounds
    assert 0.0 <= metrics["accuracy"] <= 100.0
    assert 0.0 <= metrics["precision"] <= 100.0
    assert 0.0 <= metrics["recall"] <= 100.0
    assert 0.0 <= metrics["f1"] <= 100.0
    assert 0.0 <= metrics["fpr"] <= 1.0
    assert metrics["latency_us"] > 0.0, "Latency must be strictly positive"


@pytest.mark.parametrize("model_cls", ALL_TREE_CLASSES)
def test_multiclass_fit_predict_evaluate(model_cls, synthetic_multiclass_data):
    """Test Level 2 multiclass classification pipeline with macro-averaged metrics."""
    X_train, y_train, X_test, y_test = synthetic_multiclass_data
    model = model_cls()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X_test),)

    metrics = model.evaluate(X_test, y_test)
    assert isinstance(metrics, dict)
    for k in ["accuracy", "precision", "recall", "f1", "fpr", "latency_us"]:
        assert k in metrics
    assert metrics["latency_us"] > 0.0


def test_hyperparameter_override():
    """Verify that models accept custom hyperparameters."""
    dt = DecisionTreeModel(max_depth=5, min_samples_split=20)
    assert dt.model.max_depth == 5
    assert dt.model.min_samples_split == 20

    rf = RandomForestModel(n_estimators=10, max_depth=8)
    assert rf.model.n_estimators == 10
    assert rf.model.max_depth == 8

    et = ExtraTreesModel(n_estimators=15, max_depth=6)
    assert et.model.n_estimators == 15
    assert et.model.max_depth == 6
