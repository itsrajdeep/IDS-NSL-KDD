"""
model_trainer.py — Train multiple ML models for the IDS pipeline.

Phase 9:
- Decision Tree
- Random Forest
- KNN
- Logistic Regression
- XGBoost
- Naive Bayes
"""

import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from src.config import MODEL_NAMES, RANDOM_STATE


def get_models():
    """
    Create and return a dict of model name → model instance.
    """
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            use_label_encoder=False, eval_metric='logloss',
            verbosity=0
        ),
        'Naive Bayes': GaussianNB(),
    }
    return models


def train_and_evaluate(X_train, X_test, y_train, y_test, task='binary'):
    """
    Train all 6 models and evaluate them.

    Args:
        X_train, X_test: feature arrays/DataFrames
        y_train, y_test: label arrays
        task: 'binary' or 'multiclass'

    Returns:
        results: dict of {model_name: {accuracy, precision, recall, f1, train_time, conf_matrix}}
        trained_models: dict of {model_name: fitted model}
    """
    models = get_models()

    # For multiclass XGBoost, encode string labels to integers
    label_mapping = None
    y_train_fit = y_train
    y_test_eval = y_test

    if task == 'multiclass':
        unique_labels = sorted(np.unique(np.concatenate([y_train, y_test])))
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        reverse_mapping = {idx: label for label, idx in label_mapping.items()}
        y_train_fit = np.array([label_mapping[l] for l in y_train])
        y_test_eval = np.array([label_mapping[l] for l in y_test])

    average = 'binary' if task == 'binary' else 'weighted'
    results = {}
    trained_models = {}

    for name in MODEL_NAMES:
        print(f"\n  Training: {name}...")
        model = models[name]

        # Train
        start = time.time()
        if name == 'XGBoost' and task == 'multiclass':
            model.set_params(objective='multi:softmax', num_class=len(np.unique(y_train_fit)))
            model.fit(X_train, y_train_fit)
        elif task == 'multiclass' and name not in ['Naive Bayes']:
            model.fit(X_train, y_train_fit)
        else:
            model.fit(X_train, y_train_fit)
        train_time = time.time() - start

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        acc = accuracy_score(y_test_eval, y_pred)
        prec = precision_score(y_test_eval, y_pred, average=average, zero_division=0)
        rec = recall_score(y_test_eval, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_test_eval, y_pred, average=average, zero_division=0)
        cm = confusion_matrix(y_test_eval, y_pred)

        results[name] = {
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1': round(f1 * 100, 2),
            'train_time': round(train_time, 2),
            'confusion_matrix': cm,
        }
        trained_models[name] = model

        print(f"    Accuracy:  {results[name]['accuracy']}%")
        print(f"    Precision: {results[name]['precision']}%")
        print(f"    Recall:    {results[name]['recall']}%")
        print(f"    F1-Score:  {results[name]['f1']}%")
        print(f"    Time:      {results[name]['train_time']}s")

    return results, trained_models, label_mapping
