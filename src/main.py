"""
main.py — Full IDS Pipeline Orchestrator.

Runs the complete workflow:
  Dataset → Preprocessing → Binary Classification → Multi-class Classification
  → SMOTE → Feature Selection → Train Models → Compare Results → Best Model

Usage:
    python -m src.main
"""

import os
import sys
import warnings
import numpy as np
import joblib

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODELS_DIR, RESULTS_DIR, RANDOM_STATE
from src.data_loader import load_nslkdd, explore_dataset
from src.preprocessing import prepare_features_labels
from src.smote_handler import apply_smote
from src.feature_selection import apply_feature_selection, compare_feature_selection_methods
from src.model_trainer import train_and_evaluate, get_models
from src.evaluator import (
    print_results_table, save_results_csv, generate_all_plots,
    plot_all_confusion_matrices, find_best_model
)


def run_pipeline():
    """
    Execute the full IDS pipeline.
    """
    print("\n" + "=" * 70)
    print("  INTRUSION DETECTION SYSTEM - FULL PIPELINE")
    print("  Using NSL-KDD Dataset")
    print("=" * 70)

    # Ensure output dirs exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # =========================================================================
    # PHASE 1-2: Load and Explore Dataset
    # =========================================================================
    train_df, test_df = load_nslkdd()
    explore_dataset(train_df, test_df)

    # =========================================================================
    # PHASE 5: BINARY CLASSIFICATION
    # =========================================================================
    print("\n\n" + "#" * 70)
    print("#  LEVEL 1: BINARY CLASSIFICATION (Normal vs Attack)")
    print("#" * 70)

    X_train_bin, X_test_bin, y_train_bin, y_test_bin, scaler_bin, enc_bin = \
        prepare_features_labels(train_df, test_df, task='binary')

    print("\n  --- Training 6 Models (Binary, Baseline) ---")
    results_bin_baseline, models_bin_baseline, _ = \
        train_and_evaluate(X_train_bin, X_test_bin, y_train_bin, y_test_bin, task='binary')
    print_results_table(results_bin_baseline, "Binary Classification - Baseline")

    # =========================================================================
    # PHASE 6: MULTI-CLASS CLASSIFICATION
    # =========================================================================
    print("\n\n" + "#" * 70)
    print("#  LEVEL 2: MULTI-CLASS CLASSIFICATION (Normal, DoS, Probe, R2L, U2R)")
    print("#" * 70)

    X_train_mc, X_test_mc, y_train_mc, y_test_mc, scaler_mc, enc_mc = \
        prepare_features_labels(train_df, test_df, task='multiclass')

    # ----- Config 1: Baseline (no SMOTE, no FS) -----
    print("\n  --- Config 1: Baseline (No SMOTE, No FS) ---")
    results_baseline, models_baseline, lm_baseline = \
        train_and_evaluate(X_train_mc, X_test_mc, y_train_mc, y_test_mc, task='multiclass')
    print_results_table(results_baseline, "Multi-class - Baseline")
    plot_all_confusion_matrices(results_baseline, task='multiclass', config_name='baseline')

    # ----- Config 2: With SMOTE -----
    print("\n  --- Config 2: With SMOTE ---")
    X_train_smote, y_train_smote = apply_smote(X_train_mc, y_train_mc)
    results_smote, models_smote, lm_smote = \
        train_and_evaluate(X_train_smote, X_test_mc, y_train_smote, y_test_mc, task='multiclass')
    print_results_table(results_smote, "Multi-class - With SMOTE")
    plot_all_confusion_matrices(results_smote, task='multiclass', config_name='with_smote')

    # ----- Feature Selection Comparison -----
    print("\n  --- Comparing Feature Selection Methods ---")
    fs_results = compare_feature_selection_methods(X_train_mc, X_test_mc, y_train_mc, k=20)

    # Use RF Importance as the default best FS method
    X_train_fs, X_test_fs, selected_features = fs_results['rf_importance']

    # ----- Config 3: With FS only -----
    print("\n  --- Config 3: With Feature Selection (RF Importance, k=20) ---")
    results_fs, models_fs, lm_fs = \
        train_and_evaluate(X_train_fs, X_test_fs, y_train_mc, y_test_mc, task='multiclass')
    print_results_table(results_fs, "Multi-class - With Feature Selection")
    plot_all_confusion_matrices(results_fs, task='multiclass', config_name='with_fs')

    # ----- Config 4: SMOTE + FS -----
    print("\n  --- Config 4: SMOTE + Feature Selection ---")
    X_train_smote_fs, y_train_smote_fs = apply_smote(X_train_fs, y_train_mc)
    results_smote_fs, models_smote_fs, lm_smote_fs = \
        train_and_evaluate(X_train_smote_fs, X_test_fs, y_train_smote_fs, y_test_mc, task='multiclass')
    print_results_table(results_smote_fs, "Multi-class - SMOTE + Feature Selection")
    plot_all_confusion_matrices(results_smote_fs, task='multiclass', config_name='smote_fs')

    # =========================================================================
    # PHASE 10: COMPARISON & VISUALIZATION
    # =========================================================================
    all_results = {
        'Baseline': results_baseline,
        'With SMOTE': results_smote,
        'With FS': results_fs,
        'SMOTE + FS': results_smote_fs,
    }

    # Save comparison CSV
    results_df = save_results_csv(all_results)

    # Generate all comparison plots
    generate_all_plots(all_results)

    # =========================================================================
    # PHASE 11: SELECT & SAVE BEST MODEL
    # =========================================================================
    print("\n\n" + "#" * 70)
    print("#  FINAL: BEST MODEL SELECTION")
    print("#" * 70)

    best_config, best_model_name, best_f1 = find_best_model(all_results, metric='f1')
    best_config_acc, best_model_acc, best_acc = find_best_model(all_results, metric='accuracy')

    print(f"\n  Best by F1-Score:  {best_model_name} ({best_config}) - F1: {best_f1}%")
    print(f"  Best by Accuracy:  {best_model_acc} ({best_config_acc}) - Acc: {best_acc}%")

    # Get the actual trained model
    config_models_map = {
        'Baseline': models_baseline,
        'With SMOTE': models_smote,
        'With FS': models_fs,
        'SMOTE + FS': models_smote_fs,
    }

    best_trained_model = config_models_map[best_config][best_model_name]

    # Save the best model
    model_path = os.path.join(MODELS_DIR, "ids_model.pkl")
    joblib.dump(best_trained_model, model_path)
    print(f"\n  Best model saved to: {model_path}")

    # Also save the scaler and feature info for deployment
    deployment_info = {
        'best_model_name': best_model_name,
        'best_config': best_config,
        'best_f1': best_f1,
        'best_accuracy': all_results[best_config][best_model_name]['accuracy'],
        'scaler': scaler_mc,
        'encoders': enc_mc,
        'selected_features': selected_features if 'FS' in best_config else 'all',
        'task': 'multiclass',
    }
    info_path = os.path.join(MODELS_DIR, "deployment_info.pkl")
    joblib.dump(deployment_info, info_path)
    print(f"  Deployment info saved to: {info_path}")

    # =========================================================================
    # PHASE 12: SUMMARY FOR ARCHITECTURE TEAM
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("  HANDOFF SUMMARY FOR ARCHITECTURE TEAM")
    print("=" * 70)
    print(f"""
  Model:          {best_model_name}
  Configuration:  {best_config}
  Task:           Multi-class (Normal, DoS, Probe, R2L, U2R)
  F1-Score:       {best_f1}%
  Accuracy:       {all_results[best_config][best_model_name]['accuracy']}%
  Precision:      {all_results[best_config][best_model_name]['precision']}%
  Recall:         {all_results[best_config][best_model_name]['recall']}%

  Files for deployment:
    - models/ids_model.pkl          (trained model)
    - models/deployment_info.pkl    (scaler, encoders, feature info)

  Results & Plots:
    - results/comparison_results.csv
    - results/*.png (comparison charts, confusion matrices)

  Architecture Integration:
    Edge -> Fog -> Cloud deployment ready.
    Load model with: joblib.load('models/ids_model.pkl')
""")

    print("=" * 70)
    print("  PIPELINE COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    run_pipeline()
