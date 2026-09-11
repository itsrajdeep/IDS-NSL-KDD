"""
boosting_models.py — Gradient Boosting Models for Hierarchical IDS.

Owner: lemonkartikeya
Assigned Models:
  1. LightGBMModel:  Fast, memory-efficient gradient boosting with leaf-wise growth.
  2. XGBoostModel:   High-accuracy gradient boosting with level-wise growth and regularization.
  3. CatBoostModel:  Gradient boosting with native categorical feature support and robust defaults.

All models inherit from IDSModelMixin (src.models.base) and conform to BaseIDSModel (src.contracts).
"""

from typing import Optional, Dict, Any
import numpy as np
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from src.config import RANDOM_STATE
from src.models.base import IDSModelMixin


class LightGBMModel(IDSModelMixin):
    """
    LightGBM Gradient Boosting Classifier for Intrusion Detection.

    Role in Hierarchy:
      - Leaf-wise tree growth makes it faster and more accurate than level-wise methods.
      - Strong candidate for Level 2 Cloud threat classification — excellent on tabular data.
      - Handles class imbalance via is_unbalance=True, avoiding overfit on majority class.
    """
    name: str = "LightGBM"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = -1,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        is_unbalance: bool = True,
        random_state: int = RANDOM_STATE,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.is_unbalance = is_unbalance
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.extra_kwargs = kwargs

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            is_unbalance=self.is_unbalance,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=-1,
            **self.extra_kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit LightGBM classifier on preprocessed features X and labels y."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input samples X."""
        return self.model.predict(X)


class XGBoostModel(IDSModelMixin):
    """
    XGBoost Gradient Boosting Classifier for Intrusion Detection.

    Role in Hierarchy:
      - Level-wise tree growth with L1/L2 regularization prevents overfitting.
      - Excellent for Level 2 Cloud threat classification with high recall on rare attacks.
      - eval_metric set to avoid warnings; n_jobs=-1 uses all cores.
    """
    name: str = "XGBoost"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        min_child_weight: int = 5,
        gamma: float = 0.1,
        random_state: int = RANDOM_STATE,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.extra_kwargs = kwargs
        # Internal label encoder: XGBoost requires integer labels;
        # this transparently converts string labels <-> integers.
        self._le: Optional[LabelEncoder] = None

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric="logloss",
            verbosity=0,
            **self.extra_kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit XGBoost classifier on preprocessed features X and labels y.

        XGBoost requires integer labels internally. If string labels are passed
        (e.g. 'DoS', 'Probe'), a LabelEncoder is fitted and y is encoded to ints.
        """
        if y.dtype.kind in ('U', 'S', 'O'):  # string / object dtype
            self._le = LabelEncoder()
            y = self._le.fit_transform(y)
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input samples X. Decodes back to original string labels if needed."""
        preds = self.model.predict(X)
        if self._le is not None:
            preds = self._le.inverse_transform(preds)
        return preds


class CatBoostModel(IDSModelMixin):
    """
    CatBoost Gradient Boosting Classifier for Intrusion Detection.

    Role in Hierarchy:
      - Symmetric (oblivious) trees and ordered boosting make it robust to overfitting.
      - Native support for categorical features without manual encoding.
      - Best candidate for mixed-type NSL-KDD data; strong on rare-class detection (U2R, R2L).
      - auto_class_weights='Balanced' mirrors class_weight='balanced' in sklearn models.
    """
    name: str = "CatBoost"

    def __init__(
        self,
        iterations: int = 200,
        depth: int = 8,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        border_count: int = 128,
        auto_class_weights: str = "Balanced",
        random_seed: int = RANDOM_STATE,
        **kwargs
    ):
        super().__init__()
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.auto_class_weights = auto_class_weights
        self.random_seed = random_seed
        self.extra_kwargs = kwargs

        self.model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            border_count=self.border_count,
            auto_class_weights=self.auto_class_weights,
            random_seed=self.random_seed,
            verbose=0,
            **self.extra_kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit CatBoost classifier on preprocessed features X and labels y."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input samples X.

        CatBoost can return shape (n, 1) for multiclass tasks — flatten ensures
        consistent 1D output matching the BaseIDSModel contract.
        """
        return np.array(self.model.predict(X)).flatten()


__all__ = [
    "LightGBMModel",
    "XGBoostModel",
    "CatBoostModel",
]
