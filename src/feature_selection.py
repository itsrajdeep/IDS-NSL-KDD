"""
feature_selection.py — Feature selection methods for the IDS pipeline.

Phase 8 (Your Enhancement 🔥):
- Method 1: Correlation-based (drop highly correlated features)
- Method 2: SelectKBest (statistical test)
- Method 3: Random Forest Feature Importance
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from src.config import RANDOM_STATE


def correlation_based_selection(X_train, X_test, threshold=0.95):
    """
    Remove highly correlated features.
    If two features have correlation > threshold, drop one.

    Returns:
        X_train_reduced, X_test_reduced, selected_features, dropped_features
    """
    print("\n  --- Method 1: Correlation-based Feature Selection ---")

    corr_matrix = X_train.corr().abs()

    # Upper triangle mask
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation > threshold
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    print(f"    Threshold: {threshold}")
    print(f"    Features dropped: {len(to_drop)}")
    print(f"    Features remaining: {X_train.shape[1] - len(to_drop)}")

    if to_drop:
        print(f"    Dropped: {to_drop[:10]}{'...' if len(to_drop) > 10 else ''}")

    X_train_reduced = X_train.drop(columns=to_drop)
    X_test_reduced = X_test.drop(columns=to_drop)

    return X_train_reduced, X_test_reduced, list(X_train_reduced.columns), to_drop


def selectkbest_selection(X_train, X_test, y_train, k=20):
    """
    Select top-K features using ANOVA F-test (f_classif).

    Returns:
        X_train_reduced, X_test_reduced, selected_features, scores
    """
    print(f"\n  --- Method 2: SelectKBest (k={k}) ---")

    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_reduced = selector.fit_transform(X_train, y_train)
    X_test_reduced = selector.transform(X_test)

    # Get selected feature names
    mask = selector.get_support()
    selected_features = X_train.columns[mask].tolist()
    scores = selector.scores_

    print(f"    Features selected: {k}")
    print(f"    Top features: {selected_features[:10]}{'...' if len(selected_features) > 10 else ''}")

    # Convert back to DataFrame
    X_train_reduced = pd.DataFrame(X_train_reduced, columns=selected_features, index=X_train.index)
    X_test_reduced = pd.DataFrame(X_test_reduced, columns=selected_features, index=X_test.index)

    return X_train_reduced, X_test_reduced, selected_features, scores


def rf_importance_selection(X_train, X_test, y_train, k=20):
    """
    Select top-K features based on Random Forest feature importances.

    Returns:
        X_train_reduced, X_test_reduced, selected_features, importances
    """
    print(f"\n  --- Method 3: Random Forest Feature Importance (k={k}) ---")

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Select top-K
    top_k_indices = indices[:k]
    selected_features = X_train.columns[top_k_indices].tolist()

    print(f"    Features selected: {k}")
    print(f"    Top 10 features with importance:")
    for i in range(min(10, k)):
        idx = indices[i]
        print(f"      {X_train.columns[idx]:<35} {importances[idx]:.4f}")

    X_train_reduced = X_train[selected_features]
    X_test_reduced = X_test[selected_features]

    return X_train_reduced, X_test_reduced, selected_features, importances


def apply_feature_selection(X_train, X_test, y_train, method='rf_importance', k=20, corr_threshold=0.95):
    """
    Apply a feature selection method.

    Args:
        method: 'correlation', 'selectkbest', or 'rf_importance'
        k: number of features to select (for selectkbest and rf_importance)
        corr_threshold: correlation threshold (for correlation method)

    Returns:
        X_train_reduced, X_test_reduced, selected_features, extra_info
    """
    print("\n" + "=" * 70)
    print(f"FEATURE SELECTION: {method}")
    print("=" * 70)

    if method == 'correlation':
        return correlation_based_selection(X_train, X_test, threshold=corr_threshold)
    elif method == 'selectkbest':
        return selectkbest_selection(X_train, X_test, y_train, k=k)
    elif method == 'rf_importance':
        return rf_importance_selection(X_train, X_test, y_train, k=k)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'correlation', 'selectkbest', or 'rf_importance'.")


def compare_feature_selection_methods(X_train, X_test, y_train, k=20):
    """
    Run all 3 feature selection methods and return results for comparison.

    Returns:
        dict of {method_name: (X_train_reduced, X_test_reduced, selected_features)}
    """
    print("\n" + "=" * 70)
    print("COMPARING ALL FEATURE SELECTION METHODS")
    print("=" * 70)

    results = {}

    # Method 1: Correlation
    X_tr_corr, X_te_corr, feats_corr, _ = correlation_based_selection(X_train, X_test)
    results['correlation'] = (X_tr_corr, X_te_corr, feats_corr)

    # Method 2: SelectKBest
    X_tr_skb, X_te_skb, feats_skb, _ = selectkbest_selection(X_train, X_test, y_train, k=k)
    results['selectkbest'] = (X_tr_skb, X_te_skb, feats_skb)

    # Method 3: RF Importance
    X_tr_rf, X_te_rf, feats_rf, _ = rf_importance_selection(X_train, X_test, y_train, k=k)
    results['rf_importance'] = (X_tr_rf, X_te_rf, feats_rf)

    # Summary
    print(f"\n  --- Feature Selection Summary ---")
    print(f"    Original features:     {X_train.shape[1]}")
    print(f"    After Correlation:     {len(feats_corr)}")
    print(f"    After SelectKBest:     {len(feats_skb)}")
    print(f"    After RF Importance:   {len(feats_rf)}")

    # Overlap analysis
    common_skb_rf = set(feats_skb) & set(feats_rf)
    print(f"\n    Features common to SelectKBest & RF: {len(common_skb_rf)}")
    if common_skb_rf:
        print(f"    Common: {sorted(common_skb_rf)}")

    return results
