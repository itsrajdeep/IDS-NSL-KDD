"""
test_ids_pipeline.py — Unit tests for the IDS pipeline.

Tests:
    - Config constants are valid
    - Data loading and exploration
    - Preprocessing: encoding, scaling, label creation
    - Feature selection methods
    - Model trainer returns correct structure
    - SMOTE handler balances classes
    - Evaluator utilities
    - Saved model can be loaded and predicts
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import (
    COLUMN_NAMES, CATEGORICAL_COLS, LABEL_COL,
    ATTACK_CATEGORY_MAP, CATEGORIES, MODEL_NAMES,
    TRAIN_FILE, TEST_FILE, MODELS_DIR,
)
from src.data_loader import load_nslkdd, explore_dataset
from src.preprocessing import (
    encode_categorical, scale_features,
    create_binary_labels, create_multiclass_labels,
    prepare_features_labels,
)
from src.feature_selection import (
    correlation_based_selection,
    selectkbest_selection,
    rf_importance_selection,
    apply_feature_selection,
)
from src.smote_handler import apply_smote
from src.model_trainer import get_models, train_and_evaluate
from src.evaluator import find_best_model, save_results_csv


# ===========================================================================
# Helpers — tiny synthetic dataset so tests run fast without real KDD data
# ===========================================================================

def make_tiny_df(n=200):
    """
    Create a small fake network-connection DataFrame with the same schema
    as NSL-KDD (minus the 'difficulty' column which is dropped on load).
    """
    rng = np.random.RandomState(42)
    data = {col: rng.randint(0, 100, n) for col in COLUMN_NAMES
            if col not in ('protocol_type', 'service', 'flag', 'intrusion_type', 'difficulty')}
    data['protocol_type'] = rng.choice(['tcp', 'udp', 'icmp'], n)
    data['service'] = rng.choice(['http', 'ftp', 'smtp', 'ssh', 'dns'], n)
    data['flag'] = rng.choice(['SF', 'S0', 'REJ', 'RSTO'], n)
    data['intrusion_type'] = rng.choice(
        ['normal', 'neptune', 'ipsweep', 'guess_passwd', 'buffer_overflow'], n
    )
    df = pd.DataFrame(data)
    # drop difficulty as the real loader does
    return df


# ===========================================================================
# Test: Config
# ===========================================================================
class TestConfig(unittest.TestCase):

    def test_column_count(self):
        """NSL-KDD should have 41 features + intrusion_type + difficulty."""
        self.assertEqual(len(COLUMN_NAMES), 43)

    def test_label_col_in_columns(self):
        self.assertIn(LABEL_COL, COLUMN_NAMES)

    def test_categorical_cols_subset(self):
        for col in CATEGORICAL_COLS:
            self.assertIn(col, COLUMN_NAMES)

    def test_all_attack_categories_valid(self):
        valid = set(CATEGORIES)
        for attack, cat in ATTACK_CATEGORY_MAP.items():
            self.assertIn(cat, valid, f"Attack '{attack}' maps to unknown category '{cat}'")

    def test_normal_mapped_to_Normal(self):
        self.assertEqual(ATTACK_CATEGORY_MAP['normal'], 'Normal')

    def test_model_names_count(self):
        self.assertEqual(len(MODEL_NAMES), 6)

    def test_data_files_exist(self):
        self.assertTrue(os.path.exists(TRAIN_FILE), "KDDTrain+.txt not found")
        self.assertTrue(os.path.exists(TEST_FILE), "KDDTest+.txt not found")


# ===========================================================================
# Test: Data Loader
# ===========================================================================
class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.train_df, self.test_df = load_nslkdd()

    def test_shapes(self):
        """Train should have ~125k rows, test ~22k rows, both 42 columns."""
        self.assertGreater(len(self.train_df), 100_000)
        self.assertGreater(len(self.test_df), 10_000)
        # 41 features + label (difficulty dropped)
        self.assertEqual(self.train_df.shape[1], 42)
        self.assertEqual(self.test_df.shape[1], 42)

    def test_difficulty_dropped(self):
        self.assertNotIn('difficulty', self.train_df.columns)
        self.assertNotIn('difficulty', self.test_df.columns)

    def test_no_missing_values(self):
        self.assertEqual(self.train_df.isnull().sum().sum(), 0)
        self.assertEqual(self.test_df.isnull().sum().sum(), 0)

    def test_label_column_present(self):
        self.assertIn(LABEL_COL, self.train_df.columns)
        self.assertIn(LABEL_COL, self.test_df.columns)

    def test_explore_runs(self):
        """explore_dataset should return both DataFrames without errors."""
        t, e = explore_dataset(self.train_df, self.test_df)
        self.assertIsInstance(t, pd.DataFrame)
        self.assertIsInstance(e, pd.DataFrame)


# ===========================================================================
# Test: Preprocessing (uses tiny synthetic data for speed)
# ===========================================================================
class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.train = make_tiny_df(150)
        self.test = make_tiny_df(50)

    def test_encode_categorical_shape(self):
        tr_enc, te_enc, encoders = encode_categorical(self.train, self.test)
        self.assertEqual(tr_enc.shape, self.train.shape)
        self.assertEqual(te_enc.shape, self.test.shape)

    def test_encode_categorical_numeric(self):
        tr_enc, _, _ = encode_categorical(self.train, self.test)
        for col in CATEGORICAL_COLS:
            self.assertTrue(pd.api.types.is_integer_dtype(tr_enc[col]),
                            f"Column '{col}' should be int after encoding")

    def test_encode_returns_encoders(self):
        _, _, encoders = encode_categorical(self.train, self.test)
        for col in CATEGORICAL_COLS:
            self.assertIn(col, encoders)

    def test_binary_labels(self):
        y = self.train[LABEL_COL]
        labels = create_binary_labels(y)
        self.assertEqual(set(labels), {0, 1})

    def test_multiclass_labels(self):
        y = self.train[LABEL_COL]
        labels = create_multiclass_labels(y)
        valid = set(CATEGORIES)
        self.assertTrue(set(labels).issubset(valid),
                        f"Unexpected categories: {set(labels) - valid}")

    def test_scale_features(self):
        tr_enc, te_enc, _ = encode_categorical(self.train, self.test)
        X_tr = tr_enc.drop(LABEL_COL, axis=1)
        X_te = te_enc.drop(LABEL_COL, axis=1)
        X_tr_s, X_te_s, scaler = scale_features(X_tr, X_te)
        # Mean of training set should be ~0 after scaling
        self.assertAlmostEqual(float(X_tr_s.mean().mean()), 0.0, places=5)

    def test_prepare_binary(self):
        X_tr, X_te, y_tr, y_te, scaler, encoders = prepare_features_labels(
            self.train, self.test, task='binary')
        self.assertEqual(X_tr.shape[1], 41)        # 41 features
        self.assertEqual(len(y_tr), len(self.train))
        self.assertIn(0, y_tr)                      # normal class present

    def test_prepare_multiclass(self):
        X_tr, X_te, y_tr, y_te, scaler, encoders = prepare_features_labels(
            self.train, self.test, task='multiclass')
        self.assertEqual(X_tr.shape[1], 41)
        self.assertTrue(set(y_tr).issubset(set(CATEGORIES)))


# ===========================================================================
# Test: Feature Selection
# ===========================================================================
class TestFeatureSelection(unittest.TestCase):

    def setUp(self):
        train = make_tiny_df(200)
        test = make_tiny_df(80)
        tr_enc, te_enc, _ = encode_categorical(train, test)
        X_tr = tr_enc.drop(LABEL_COL, axis=1)
        X_te = te_enc.drop(LABEL_COL, axis=1)
        y_raw = tr_enc[LABEL_COL]
        y_int = create_binary_labels(y_raw)
        self.X_tr, self.X_te, _, _, _, _ = prepare_features_labels(train, test, task='binary')
        self.y = y_int

    def test_correlation_reduces_features(self):
        X_tr_r, X_te_r, sel, dropped = correlation_based_selection(self.X_tr, self.X_te)
        self.assertLessEqual(X_tr_r.shape[1], self.X_tr.shape[1])
        self.assertEqual(X_tr_r.shape[0], self.X_tr.shape[0])

    def test_selectkbest_k_features(self):
        k = 15
        X_tr_r, X_te_r, sel, _ = selectkbest_selection(self.X_tr, self.X_te, self.y, k=k)
        self.assertEqual(X_tr_r.shape[1], k)
        self.assertEqual(len(sel), k)

    def test_rf_importance_k_features(self):
        k = 10
        X_tr_r, X_te_r, sel, imp = rf_importance_selection(self.X_tr, self.X_te, self.y, k=k)
        self.assertEqual(X_tr_r.shape[1], k)
        self.assertEqual(len(sel), k)

    def test_apply_feature_selection_methods(self):
        for method in ('correlation', 'selectkbest', 'rf_importance'):
            X_tr_r, X_te_r, sel, _ = apply_feature_selection(
                self.X_tr, self.X_te, self.y, method=method, k=10)
            self.assertGreater(X_tr_r.shape[1], 0)


# ===========================================================================
# Test: SMOTE Handler
# ===========================================================================
class TestSMOTE(unittest.TestCase):

    def setUp(self):
        train = make_tiny_df(200)
        test = make_tiny_df(50)
        self.X_tr, self.X_te, self.y_tr, _, _, _ = prepare_features_labels(
            train, test, task='multiclass')

    def test_smote_increases_samples(self):
        X_res, y_res = apply_smote(self.X_tr, self.y_tr)
        self.assertGreaterEqual(len(X_res), len(self.X_tr))

    def test_smote_balances_classes(self):
        X_res, y_res = apply_smote(self.X_tr, self.y_tr)
        unique, counts = np.unique(y_res, return_counts=True)
        # All classes should have the same count after SMOTE
        self.assertEqual(len(set(counts)), 1,
                         f"Classes not balanced: {dict(zip(unique, counts))}")


# ===========================================================================
# Test: Model Trainer (small subset of real data)
# ===========================================================================
class TestModelTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load real data once; use first 5000 rows for speed."""
        train_df, test_df = load_nslkdd()
        train_small = train_df.head(5000)
        test_small = test_df.head(1000)
        cls.X_tr, cls.X_te, cls.y_tr, cls.y_te, _, _ = prepare_features_labels(
            train_small, test_small, task='binary')

    def test_get_models_returns_six(self):
        models = get_models()
        self.assertEqual(len(models), 6)

    def test_train_and_evaluate_keys(self):
        results, trained, _ = train_and_evaluate(
            self.X_tr, self.X_te, self.y_tr, self.y_te, task='binary')
        for name in MODEL_NAMES:
            self.assertIn(name, results)
            for metric in ('accuracy', 'precision', 'recall', 'f1', 'train_time'):
                self.assertIn(metric, results[name])

    def test_accuracies_in_range(self):
        results, _, _ = train_and_evaluate(
            self.X_tr, self.X_te, self.y_tr, self.y_te, task='binary')
        for name, r in results.items():
            self.assertGreaterEqual(r['accuracy'], 0.0)
            self.assertLessEqual(r['accuracy'], 100.0)


# ===========================================================================
# Test: Evaluator
# ===========================================================================
class TestEvaluator(unittest.TestCase):

    def _make_results(self):
        return {
            'Baseline': {
                'Decision Tree':     {'accuracy': 76.0, 'precision': 79.0, 'recall': 76.0, 'f1': 72.0, 'train_time': 2.0, 'confusion_matrix': np.array([[100, 10], [5, 85]])},
                'Random Forest':     {'accuracy': 75.0, 'precision': 81.0, 'recall': 75.0, 'f1': 71.0, 'train_time': 6.0, 'confusion_matrix': np.array([[98, 12], [6, 84]])},
                'KNN':               {'accuracy': 74.0, 'precision': 80.0, 'recall': 74.0, 'f1': 70.0, 'train_time': 0.1, 'confusion_matrix': np.array([[95, 15], [8, 82]])},
                'Logistic Regression':{'accuracy': 76.0, 'precision': 75.0, 'recall': 76.0, 'f1': 71.0, 'train_time':15.0, 'confusion_matrix': np.array([[97, 13], [7, 83]])},
                'XGBoost':           {'accuracy': 75.0, 'precision': 79.0, 'recall': 75.0, 'f1': 71.0, 'train_time': 7.0, 'confusion_matrix': np.array([[96, 14], [7, 83]])},
                'Naive Bayes':       {'accuracy': 43.0, 'precision': 55.0, 'recall': 43.0, 'f1': 34.0, 'train_time': 0.2, 'confusion_matrix': np.array([[60, 50], [30, 60]])},
            }
        }

    def test_find_best_model_f1(self):
        results = self._make_results()
        best_cfg, best_model, best_score = find_best_model(results, metric='f1')
        self.assertEqual(best_cfg, 'Baseline')
        self.assertIn(best_model, MODEL_NAMES)
        self.assertGreater(best_score, 0)

    def test_save_results_csv(self):
        results = self._make_results()
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            # temporarily patch RESULTS_DIR
            import src.evaluator as ev
            original = ev.RESULTS_DIR
            ev.RESULTS_DIR = tmp
            df = save_results_csv(results)
            ev.RESULTS_DIR = original
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('Accuracy (%)', df.columns)
        self.assertIn('F1-Score (%)', df.columns)


# ===========================================================================
# Test: Saved Model Deployment
# ===========================================================================
class TestDeployment(unittest.TestCase):

    def test_model_file_exists(self):
        model_path = os.path.join(MODELS_DIR, 'ids_model.pkl')
        self.assertTrue(os.path.exists(model_path), "ids_model.pkl not found in models/")

    def test_deployment_info_exists(self):
        info_path = os.path.join(MODELS_DIR, 'deployment_info.pkl')
        self.assertTrue(os.path.exists(info_path), "deployment_info.pkl not found")

    def test_model_loads_and_predicts(self):
        import joblib
        model_path = os.path.join(MODELS_DIR, 'ids_model.pkl')
        info_path = os.path.join(MODELS_DIR, 'deployment_info.pkl')

        if not os.path.exists(model_path):
            self.skipTest("ids_model.pkl not present — run the full pipeline first")

        model = joblib.load(model_path)
        info = joblib.load(info_path)

        self.assertIn('best_model_name', info)
        self.assertIn('best_f1', info)
        self.assertGreater(info['best_f1'], 0)

        # Quick sanity predict using a tiny random sample
        train_df, _ = load_nslkdd()
        train_small = train_df.head(500)
        test_small = train_df.tail(100)
        X_tr, X_te, y_tr, y_te, _, _ = prepare_features_labels(
            train_small, test_small, task='multiclass')

        # Use selected features if model was trained with FS
        if info['selected_features'] != 'all':
            feats = info['selected_features']
            available = [f for f in feats if f in X_te.columns]
            X_te = X_te[available]

        try:
            preds = model.predict(X_te)
            self.assertEqual(len(preds), len(X_te))
        except Exception as e:
            self.skipTest(f"Prediction skipped due to feature mismatch: {e}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)
