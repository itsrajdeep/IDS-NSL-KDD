"""
contracts.py — The shared contract every IDS model MUST follow.

FROZEN: Do not modify after initial setup.
Both tree_models.py (4kub0) and boosting_models.py (lemonkartikeya)
depend on this file. Changing it breaks both.

WHY THIS FILE EXISTS:
    Without a contract, each dev writes their model differently.
    4kub0 might call his method .train(), you might call yours .fit().
    The benchmark runner (evaluator.py) wouldn't know how to call either.
    This file forces both of you to speak the same language.
"""

# abc = Abstract Base Classes — Python's built-in way to define interfaces.
# ABC is the base class that makes a class "abstract" (can't be instantiated directly).
# abstractmethod is a decorator that marks a method as "you MUST implement this".
from abc import ABC, abstractmethod

# We type-hint with numpy arrays so both devs know exactly what to pass in/out.
import numpy as np


class BaseIDSModel(ABC):
    """
    Abstract contract that every IDS model must implement.

    WHY ABSTRACT?
        If you try to do:  model = BaseIDSModel()   <-- Python raises a TypeError.
        You CANNOT use this class directly. You must create a subclass
        (like LightGBMModel) and implement every @abstractmethod in it.
        This forces both devs to implement all required methods.
    """

    # -------------------------------------------------------------------
    # CLASS ATTRIBUTE: name
    # -------------------------------------------------------------------
    # Every model must declare its own name as a class-level string.
    # Example in your file:  name = "LightGBM"
    # The registry uses this to identify and look up models by name.
    # It's a class attribute (not a method) so it can be read without
    # instantiating the model — useful for listing available models.
    name: str = ""

    # -------------------------------------------------------------------
    # ABSTRACT METHOD: fit
    # -------------------------------------------------------------------
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the given data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
               Already preprocessed — scaled, encoded, SMOTE-balanced.
            y: Label array of shape (n_samples,).
               Binary (0/1) for L1, or category strings for L2.

        Returns:
            None. The model stores its learned state internally.

        WHY ABSTRACT?
            Every algorithm trains differently (trees split nodes,
            boosting builds residuals). Each dev implements their own
            version in their file. But the SIGNATURE is fixed here —
            always (X, y), always returns None.
        """
        ...  # '...' means "no body here, subclass must provide it"

    # -------------------------------------------------------------------
    # ABSTRACT METHOD: predict
    # -------------------------------------------------------------------
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for new data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted label array of shape (n_samples,).

        WHY ABSTRACT?
            Each model calls its own internal .predict(). LightGBM,
            XGBoost, CatBoost all do it differently internally.
            But from the outside, the evaluator just calls .predict(X)
            on ANY model and gets back an array. Uniform interface.
        """
        ...

    # -------------------------------------------------------------------
    # ABSTRACT METHOD: evaluate
    # -------------------------------------------------------------------
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate the model and return a standard metrics dictionary.

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            dict with EXACTLY these keys (benchmark runner depends on this):
            {
                'accuracy':   float  — % correctly classified
                'precision':  float  — % of predicted positives that are real
                'recall':     float  — % of real positives caught (CRITICAL for L1)
                'f1':         float  — harmonic mean of precision & recall
                'fpr':        float  — false positive rate (false alarms)
                'latency_us': float  — microseconds per packet (CRITICAL for L1 edge)
            }

        WHY ABSTRACT?
            The evaluate() method is actually NOT abstract in practice —
            base.py provides a ready-made implementation via IDSModelMixin.
            You'll inherit from IDSModelMixin, not BaseIDSModel directly.
            This signature here just documents the contract.
        """
        ...
