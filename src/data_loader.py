"""
data_loader.py — Load and explore the NSL-KDD dataset.

Phase 2-3 of the IDS pipeline:
- Load KDDTrain+.txt and KDDTest+.txt
- Drop the difficulty column
- Print dataset statistics
"""

import pandas as pd
import numpy as np
from src.config import COLUMN_NAMES, LABEL_COL, TRAIN_FILE, TEST_FILE


def load_nslkdd(train_path=None, test_path=None):
    """
    Load NSL-KDD train and test datasets.

    Returns:
        train_df, test_df: pandas DataFrames
    """
    train_path = train_path or TRAIN_FILE
    test_path = test_path or TEST_FILE

    print("=" * 70)
    print("PHASE 1: Loading NSL-KDD Dataset")
    print("=" * 70)

    train_df = pd.read_csv(train_path, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, names=COLUMN_NAMES)

    # Drop difficulty column (not useful for ML)
    train_df.drop('difficulty', axis=1, inplace=True)
    test_df.drop('difficulty', axis=1, inplace=True)

    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape:  {test_df.shape}")

    return train_df, test_df


def explore_dataset(train_df, test_df):
    """
    Print dataset statistics: missing values, duplicates, class distribution.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Dataset Exploration")
    print("=" * 70)

    # Missing values
    train_missing = train_df.isnull().sum().sum()
    test_missing = test_df.isnull().sum().sum()
    print(f"\n  Missing values (train): {train_missing}")
    print(f"  Missing values (test):  {test_missing}")

    # Duplicates
    train_dupes = train_df.duplicated().sum()
    test_dupes = test_df.duplicated().sum()
    print(f"\n  Duplicate rows (train): {train_dupes}")
    print(f"  Duplicate rows (test):  {test_dupes}")

    # Class distribution
    print(f"\n  --- Train Label Distribution ---")
    train_dist = train_df[LABEL_COL].value_counts()
    for label, count in train_dist.items():
        pct = round(count / len(train_df) * 100, 2)
        print(f"    {label:<25} {count:>6}  ({pct}%)")

    print(f"\n  --- Test Label Distribution ---")
    test_dist = test_df[LABEL_COL].value_counts()
    for label, count in test_dist.items():
        pct = round(count / len(test_df) * 100, 2)
        print(f"    {label:<25} {count:>6}  ({pct}%)")

    # Unique attack types
    all_labels = set(train_df[LABEL_COL].unique()) | set(test_df[LABEL_COL].unique())
    print(f"\n  Total unique attack types: {len(all_labels)}")
    print(f"  Labels: {sorted(all_labels)}")

    return train_df, test_df
