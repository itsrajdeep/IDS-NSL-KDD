# Intrusion Detection System (IDS)

An intelligent Intrusion Detection System built using the **NSL-KDD** dataset. This project compares multiple machine learning models across different configurations to find the optimal IDS classifier.

## Project Structure

```
Intrusion-Detection-System/
├── data/
│   ├── KDDTrain+.txt          # NSL-KDD training data
│   └── KDDTest+.txt           # NSL-KDD testing data
├── models/                    # Saved best model & deployment info
├── results/                   # Comparison tables, plots, confusion matrices
├── src/
│   ├── __init__.py
│   ├── config.py              # Column names, attack mappings, paths
│   ├── data_loader.py         # Load & explore NSL-KDD
│   ├── preprocessing.py       # LabelEncoder, StandardScaler, label creation
│   ├── smote_handler.py       # SMOTE oversampling for imbalanced classes
│   ├── feature_selection.py   # 3 feature selection methods
│   ├── model_trainer.py       # Train 6 ML algorithms
│   ├── evaluator.py           # Metrics, plots, comparison tables
│   └── main.py                # Full pipeline orchestrator
├── requirements.txt
├── .gitignore
└── README.md
```

## Models Compared

| # | Model               | Type             |
|---|---------------------|------------------|
| 1 | Decision Tree       | Tree-based       |
| 2 | Random Forest       | Ensemble         |
| 3 | KNN                 | Instance-based   |
| 4 | Logistic Regression | Linear           |
| 5 | XGBoost             | Gradient Boosting|
| 6 | Naive Bayes         | Probabilistic    |

## Configurations

Each model is evaluated under 4 configurations:

1. **Baseline** - No SMOTE, no feature selection
2. **With SMOTE** - SMOTE oversampling to balance classes
3. **With Feature Selection** - RF Importance (top 20 features)
4. **SMOTE + Feature Selection** - Both techniques combined

## Feature Selection Methods

Three methods are compared:

1. **Correlation-based** - Drop features with correlation > 0.95
2. **SelectKBest** - ANOVA F-test, top-K features
3. **Random Forest Feature Importance** - Top-K by RF importance scores

## Classification Tasks

- **Binary**: Normal (0) vs Attack (1)
- **Multi-class**: Normal, DoS, Probe, R2L, U2R

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python -m src.main
```

## Output

- `models/ids_model.pkl` - Best trained model
- `models/deployment_info.pkl` - Scaler, encoders, feature info for deployment
- `results/comparison_results.csv` - Full comparison table
- `results/*.png` - Comparison bar charts and confusion matrix heatmaps

## Deployment

```python
import joblib

# Load the best model
model = joblib.load('models/ids_model.pkl')
info = joblib.load('models/deployment_info.pkl')

print(f"Best model: {info['best_model_name']}")
print(f"Configuration: {info['best_config']}")
print(f"F1-Score: {info['best_f1']}%")
```

## Dataset

**NSL-KDD** - An improved version of the original KDD Cup 1999 dataset:
- 41 features per network connection
- 5 classes: Normal, DoS, Probe, R2L, U2R
- Training samples: ~125,973
- Test samples: ~22,544
