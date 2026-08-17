"""
preprocessing.py — Data preprocessing for the IDS pipeline.

Phase 4:
- Encode categorical columns (LabelEncoder)
- Scale features (StandardScaler)
- Create binary labels (Normal=0, Attack=1)
- Create multi-class labels (Normal, DoS, Probe, R2L, U2R)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.config import CATEGORICAL_COLS, LABEL_COL, ATTACK_CATEGORY_MAP, RANDOM_STATE


def encode_categorical(train_df, test_df):
    """
    Encode categorical columns using LabelEncoder.
    Fits on train, transforms both train and test.

    Returns:
        train_df, test_df with encoded categoricals
        encoders: dict of {col_name: fitted LabelEncoder}
    """
    print("\n" + "=" * 70)
    print("PHASE 3: Encoding Categorical Features")
    print("=" * 70)

    encoders = {}
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        # Fit on combined unique values (handles unseen labels in test)
        combined_values = pd.concat([train_df[col], test_df[col]]).unique()
        le.fit(combined_values)

        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        encoders[col] = le

        print(f"  Encoded '{col}': {len(le.classes_)} unique values")

    return train_df, test_df, encoders


def scale_features(X_train, X_test):
    """
    Normalize features using StandardScaler.
    Fits on X_train, transforms both.

    Returns:
        X_train_scaled, X_test_scaled (as DataFrames), scaler
    """
    print("\n  Scaling features with StandardScaler...")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    print(f"  Scaled {X_train_scaled.shape[1]} features.")
    return X_train_scaled, X_test_scaled, scaler


def create_binary_labels(y):
    """
    Convert intrusion_type labels to binary:
        normal → 0
        any attack → 1

    Returns:
        numpy array of 0/1
    """
    return (y != 'normal').astype(int).values


def create_multiclass_labels(y):
    """
    Map intrusion_type labels to 5 attack categories:
        Normal, DoS, Probe, R2L, U2R

    Unknown attack types default to the label itself (shouldn't happen
    with complete mapping).

    Returns:
        numpy array of category strings
    """
    return y.map(lambda x: ATTACK_CATEGORY_MAP.get(x, x)).values


def prepare_features_labels(train_df, test_df, task='binary'):
    """
    Full preprocessing pipeline:
    1. Encode categorical features
    2. Separate features (X) and labels (y)
    3. Create binary or multi-class labels
    4. Scale features

    Args:
        train_df, test_df: raw DataFrames
        task: 'binary' or 'multiclass'

    Returns:
        X_train, X_test, y_train, y_test, scaler, encoders
    """
    print(f"\n  Preparing data for {'Binary' if task == 'binary' else 'Multi-class'} classification...")

    # Step 1: Encode categoricals
    train_enc, test_enc, encoders = encode_categorical(train_df, test_df)

    # Step 2: Separate features and labels
    X_train = train_enc.drop(LABEL_COL, axis=1)
    X_test = test_enc.drop(LABEL_COL, axis=1)
    y_train_raw = train_enc[LABEL_COL]
    y_test_raw = test_enc[LABEL_COL]

    # Step 3: Create labels
    if task == 'binary':
        y_train = create_binary_labels(y_train_raw)
        y_test = create_binary_labels(y_test_raw)
        print(f"\n  Binary labels - Train: Normal={np.sum(y_train==0)}, Attack={np.sum(y_train==1)}")
        print(f"  Binary labels - Test:  Normal={np.sum(y_test==0)}, Attack={np.sum(y_test==1)}")
    else:
        y_train = create_multiclass_labels(y_train_raw)
        y_test = create_multiclass_labels(y_test_raw)
        print(f"\n  Multi-class labels (train):")
        for cat in np.unique(y_train):
            print(f"    {cat:<10} {np.sum(y_train == cat):>6}")
        print(f"  Multi-class labels (test):")
        for cat in np.unique(y_test):
            print(f"    {cat:<10} {np.sum(y_test == cat):>6}")

    # Step 4: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders
