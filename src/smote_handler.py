"""
smote_handler.py — SMOTE oversampling for imbalanced datasets.

Phase 7:
- Apply SMOTE to balance training data
- Print before/after class distributions
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from src.config import RANDOM_STATE


def apply_smote(X_train, y_train):
    """
    Apply SMOTE to balance the training dataset.

    Args:
        X_train: feature DataFrame/array
        y_train: label array

    Returns:
        X_resampled, y_resampled
    """
    print("\n" + "=" * 70)
    print("SMOTE: Balancing Training Data")
    print("=" * 70)

    # Before SMOTE
    unique, counts = np.unique(y_train, return_counts=True)
    print("\n  Before SMOTE:")
    for cls, cnt in zip(unique, counts):
        print(f"    {str(cls):<10} {cnt:>6}")
    print(f"    Total:     {len(y_train):>6}")

    # Apply SMOTE
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # After SMOTE
    unique, counts = np.unique(y_resampled, return_counts=True)
    print("\n  After SMOTE:")
    for cls, cnt in zip(unique, counts):
        print(f"    {str(cls):<10} {cnt:>6}")
    print(f"    Total:     {len(y_resampled):>6}")

    return X_resampled, y_resampled
