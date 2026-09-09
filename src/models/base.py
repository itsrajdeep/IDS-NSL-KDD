"""
base.py — Shared base class with evaluate() already implemented.

FROZEN: Do not modify after initial setup.

WHY THIS FILE EXISTS:
    contracts.py declared WHAT evaluate() must return.
    This file actually BUILDS evaluate() so neither dev has to write it.

    Without this, both 4kub0 and you would each write your own evaluate()
    function, probably slightly differently — different FPR formula,
    different latency measurement, etc. The benchmark results would be
    incomparable. This file ensures both of you measure IDENTICALLY.

HOW TO USE:
    Instead of inheriting from BaseIDSModel directly, inherit from
    IDSModelMixin:

        class LightGBMModel(IDSModelMixin):   <-- correct
            name = "LightGBM"
            def fit(self, X, y): ...
            def predict(self, X): ...
            # evaluate() is FREE — inherited from here, no need to write it
"""

import time                     # For measuring inference latency in microseconds
import numpy as np              # Array math (FPR calculation uses matrix ops)

# sklearn metrics — the standard evaluation toolkit
from sklearn.metrics import (
    accuracy_score,             # (correct predictions) / (total predictions)
    precision_score,            # (true positives) / (true positives + false positives)
    recall_score,               # (true positives) / (true positives + false negatives)
    f1_score,                   # 2 * (precision * recall) / (precision + recall)
    confusion_matrix,           # Matrix of actual vs predicted — needed for FPR
)

# Import the contract — IDSModelMixin extends it with a concrete evaluate()
from src.contracts import BaseIDSModel


class IDSModelMixin(BaseIDSModel):
    """
    Mixin that provides a ready-made evaluate() to any model class.

    WHY 'MIXIN'?
        A mixin is a class that adds behaviour but isn't meant to be
        used on its own. It's the middle layer:
            BaseIDSModel (defines WHAT)
                └── IDSModelMixin (implements HOW for evaluate)
                        └── LightGBMModel / DecisionTreeModel (implements fit + predict)

    WHAT STAYS ABSTRACT?
        fit() and predict() are still abstract here — each model file
        must implement them. Only evaluate() is filled in.
    """

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate the trained model. Called by the benchmark runner on every model.
        Returns the standard metrics dict defined in contracts.py.
        """

        # -----------------------------------------------------------------
        # STEP 1: Measure per-packet inference latency
        # -----------------------------------------------------------------
        # perf_counter() is a high-resolution timer — much more precise
        # than time.time() for microsecond-level measurements.
        t0 = time.perf_counter()

        # Actually run the model on the test set.
        # self.predict() calls the subclass's own predict() method.
        preds = self.predict(X)

        t1 = time.perf_counter()

        # Total time / number of samples = time per sample (seconds)
        # Multiply by 1e6 to convert seconds → microseconds
        # L1 edge filter must be ≤ 50 µs/packet — this is how we check that.
        latency_us = (t1 - t0) / len(X) * 1e6

        # -----------------------------------------------------------------
        # STEP 2: Confusion matrix for FPR
        # -----------------------------------------------------------------
        # confusion_matrix(true, predicted) gives a matrix where:
        #   cm[i][j] = number of samples with true class i predicted as class j
        #
        # For binary:          For multiclass (per-class):
        #   [[TN, FP],             each row is one class
        #    [FN, TP]]
        cm = confusion_matrix(y, preds)

        # np.diag(cm) = correctly classified counts (True Positives per class)
        # cm.sum(axis=0) = total predicted as each class
        # FP per class = predicted as this class - correctly predicted as this class
        fp = cm.sum(axis=0) - np.diag(cm)

        # TN per class = everything NOT in this class's row or column + correct
        tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))

        # FPR = FP / (FP + TN)  — "of all real negatives, how many did we falsely flag?"
        # + 1e-9 prevents division by zero (if a class has no negatives)
        # np.mean() averages FPR across all classes (macro average)
        fpr = float(np.mean(fp / (fp + tn + 1e-9)))

        # -----------------------------------------------------------------
        # STEP 3: Standard classification metrics
        # -----------------------------------------------------------------
        # Determine averaging strategy based on task:
        #   'binary'  — L1: Normal vs. Attack  (2 classes)
        #   'macro'   — L2: DoS/Probe/R2L/U2R  (4+ classes, equal weight per class)
        #
        # Macro is important for L2 because U2R has only 52 samples vs DoS's 45,927.
        # Macro gives equal weight to each class regardless of size,
        # so rare attack types like U2R aren't drowned out by DoS's dominance.
        unique_classes = len(np.unique(y))
        avg = 'binary' if unique_classes == 2 else 'macro'

        return {
            # Multiply by 100 to express as percentage (e.g. 99.7 not 0.997)
            # round(..., 4) gives 4 decimal places — enough for Pareto plot precision
            'accuracy':   round(accuracy_score(y, preds) * 100, 4),
            'precision':  round(precision_score(y, preds, average=avg, zero_division=0) * 100, 4),
            'recall':     round(recall_score(y, preds, average=avg, zero_division=0) * 100, 4),
            'f1':         round(f1_score(y, preds, average=avg, zero_division=0) * 100, 4),
            'fpr':        round(fpr, 6),
            'latency_us': round(latency_us, 4),
        }
