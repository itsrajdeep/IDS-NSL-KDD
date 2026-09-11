"""
tree_models.py — Tree & Instance Ensembles for Hierarchical IDS.

Owner: 4kub0
Assigned Models:
  1. DecisionTreeModel: Fast, interpretable single-tree classifier.
  2. RandomForestModel: Variance-reduced bagging ensemble of decision trees.
  3. ExtraTreesModel: Extremely Randomized Trees for fast training/inference.
  4. BaselineNBModel: Gaussian Naive Bayes probabilistic baseline.

All models inherit from IDSModelMixin (src.models.base) and conform to BaseIDSModel (src.contracts).
"""

from typing import Optional, Union, Dict, Any
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB

from src.config import RANDOM_STATE
from src.models.base import IDSModelMixin


class DecisionTreeModel(IDSModelMixin):
    """
    Single Decision Tree Classifier for Intrusion Detection.

    Role in Hierarchy:
      - Prime candidate for Level 1 Edge filter: microsecond latency (2-5 µs),
        tiny memory footprint (< 100 KB), and directly exportable to firewall rules.
    """
    name: str = "DecisionTree"

    def __init__(
        self,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        class_weight: Optional[Union[str, Dict[Any, float]]] = "balanced",
        random_state: int = RANDOM_STATE,
        **kwargs
    ):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.extra_kwargs = kwargs

        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            **self.extra_kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit decision tree on preprocessed features X and labels y."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input samples X."""
        return self.model.predict(X)


class RandomForestModel(IDSModelMixin):
    """
    Random Forest Ensemble Classifier for Intrusion Detection.

    Role in Hierarchy:
      - Robust ensemble of B trees trained in parallel with bootstrap aggregation (bagging).
      - Excellent candidate for Level 2 Cloud threat classification with high generalization.
    """
    name: str = "RandomForest"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 20,
        min_samples_split: int = 10,
        min_samples_leaf: int = 2,
        class_weight: Optional[Union[str, Dict[Any, float]]] = "balanced",
        random_state: int = RANDOM_STATE,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.extra_kwargs = kwargs

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **self.extra_kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit random forest ensemble on features X and labels y."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input samples X."""
        return self.model.predict(X)


class ExtraTreesModel(IDSModelMixin):
    """
    Extremely Randomized Trees (Extra Trees) Classifier.

    Role in Hierarchy:
      - Uses completely random split thresholds rather than optimal cuts.
      - Drastically speeds up training & evaluation while acting as strong regularizer.
      - Top contender for both Level 1 Edge filter and Level 2 Cloud analysis.
    """
    name: str = "ExtraTrees"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 20,
        min_samples_split: int = 10,
        min_samples_leaf: int = 2,
        class_weight: Optional[Union[str, Dict[Any, float]]] = "balanced",
        random_state: int = RANDOM_STATE,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.extra_kwargs = kwargs

        self.model = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **self.extra_kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit extra trees ensemble on features X and labels y."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input samples X."""
        return self.model.predict(X)


class BaselineNBModel(IDSModelMixin):
    """
    Gaussian Naive Bayes Baseline Classifier.

    Role in Hierarchy:
      - Fast probabilistic baseline with zero hyperparameter tuning.
      - Evaluates the lower bound of performance in ablation studies.
    """
    name: str = "NaiveBayes"

    def __init__(self, var_smoothing: float = 1e-9, **kwargs):
        super().__init__()
        self.var_smoothing = var_smoothing
        self.extra_kwargs = kwargs

        self.model = GaussianNB(
            var_smoothing=self.var_smoothing,
            **self.extra_kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Gaussian Naive Bayes on features X and labels y."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input samples X."""
        return self.model.predict(X)


__all__ = [
    "DecisionTreeModel",
    "RandomForestModel",
    "ExtraTreesModel",
    "BaselineNBModel",
]
