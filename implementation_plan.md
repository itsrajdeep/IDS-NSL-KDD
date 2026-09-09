# Hierarchical IDS with Fog-Cloud Simulation: Project Pipeline & Implementation Plan

This document establishes the architecture and collaboration plan for the Hierarchical Intrusion Detection System (IDS).

---

## 1. Project Division & Ownership

| Phase | Description | Environment | Primary Owner |
|---|---|---|---|
| **Part 1: Modelling** | ML Pipeline, Preprocessing, Evolutionary Feature Selection, Hierarchical Model Suites & Benchmark Matrix | Python (`scikit-learn`, `EvoloPy`, `lightgbm`, `xgboost`, `catboost`) | **`4kub0` & `lemonkartikeya`** |
| **Part 2: Simulation & Deployment** | Network Topology Design, Edge/Fog/Cloud Node Deployment, Fault-Tolerance & Failover Simulation | Java (`iFogSim` / Maven) | **Collaborator 3 (Separate Team Member)** |

---

## 2. Part 1: Modelling (ML Pipeline & Evaluation)

Part 1 focuses on developing the hierarchical intelligence of the system. 

### 2.1 Data Preprocessing & Balancing Module
- **Data Ingestion & Cleaning**: Load NSL-KDD (and future datasets), impute missing values with medians, replace infinities, drop duplicates and zero-variance constant columns (`num_outbound_cmds`).
- **Encoding & Scaling**: Label encoding for categorical fields (`protocol_type`, `service`, `flag`). Min-Max scaling to $[0, 1]$ fit on training split only (preventing test set data leakage).
- **Imbalance Handling**:
  - **Level 1 (Binary Filter)**: Normal vs. Attack.
  - **Level 2 (Multiclass Categorizer)**: Extreme imbalance handling (DoS: 45,927 vs. U2R: 52). Uses **Capped SMOTE** with dynamic $k$-neighbors ($k = \min(5, n_{samples} - 1)$) and a ceiling multiplier to prevent artificial boundary hallucination.
  - **Artifacts**: Saved to `data/processed/level1_binary.npz`, `data/processed/level2_multiclass.npz`, `transformers.pkl`.

### 2.2 Evolutionary Feature Selection (EvoloPy)
- **Metaheuristics Suite**: 6 optimization algorithms (GA, PSO, GWO, WOA, MFO, MVO).
- **Wrapper Fitness Function**: Evaluates candidate subsets using continuous positions thresholded at $0.5$ on an inner stratified validation split:
  $$\text{fitness} = \alpha \cdot (1 - F_1) + \beta \cdot \left(\frac{k}{n}\right) \quad (\alpha = 0.99, \beta = 0.01)$$
- **Outputs**: Top feature subsets promoted per tier, exported to `data/processed/feature_subsets.pkl`.

### 2.3 Hierarchical ML Engine Development
- **Level 1 (Edge Binary Filter)**: Classifies Normal vs. Attack on binary feature subsets. Optimized for $\ge 99.5\%$ Recall and microsecond inference latency ($\le 50 \mu s$).
- **Level 2 (Cloud Multiclass Categorizer)**: Classifies specific attack families (DoS, Probe, R2L, U2R) on multiclass feature subsets. Evaluated on traffic flagged by Level 1 to reflect true cascade performance.
- **Model Suite**: Decision Trees (DT), Random Forest (RF), Extra Trees (ET), LightGBM (LGBM), XGBoost (XGB), and CatBoost.

### 2.4 Experimental Validation & Evaluation Matrix
- **Matrix**: 36 experiments per dataset ($6 \text{ feature subsets} \times 6 \text{ algorithms}$).
- **Evaluation Metrics**: Accuracy, Precision, Recall, Macro-$F_1$, False Positive Rate (FPR), and per-packet Inference Latency ($\mu s$).
- **Reporting**: Comparative tables (`results/comparison_results.csv`), ROC curves, confusion matrices, and Latency vs. $F_1$ Pareto frontier plots.

---

## 3. Collaborative Split for Part 1 (Between `4kub0` & `lemonkartikeya`)

To ensure **both developers work on and learn machine learning**, work is divided by **model families** while sharing standard contracts:

| Track | Owner | Models Assigned | Learning Focus | Exclusive Source Files |
|---|---|---|---|---|
| **Tree & Instance Ensembles** | **`4kub0`** | • **Decision Trees** & **Random Forest**<br>• **Extra Trees (Extremely Randomized Trees)**<br>• **Baseline**: Naive Bayes / KNN | • Tree-based decision boundaries<br>• Hyperparameter tuning (`max_depth`, `min_samples_split`, `class_weight`)<br>• Microsecond latency profiling | `src/models/tree_models.py`<br>`tests/test_tree_models.py` |
| **Gradient Boosted Trees** | **`lemonkartikeya`** | • **LightGBM** (histogram / leaf-wise)<br>• **XGBoost** (exact greedy / regularized)<br>• **CatBoost** (symmetric / categorical) | • Gradient boosting mechanics<br>• Residual loss minimization<br>• Multiclass loss functions on imbalanced data | `src/models/boosting_models.py`<br>`tests/test_boosting_models.py` |

---

## 4. Antigravity Agent Merge-Friendly Architecture

To prevent Git merge conflicts between both developers and their Antigravity AI agents, the codebase uses a **Plugin & Registry Pattern**:

```
IDS-NSL-KDD/
├── src/
│   ├── config.py                 # FROZEN: Common paths, columns, categories
│   ├── contracts.py              # FROZEN: BaseIDSModel abstract interface
│   ├── data/
│   │   ├── loader.py             # Data ingestion & cleaning
│   │   └── balancing.py          # Capped SMOTE & ADASYN
│   ├── features/
│   │   └── evolopy_wrapper.py    # Evolutionary feature search (GA, PSO, GWO)
│   ├── models/
│   │   ├── __init__.py           # Dynamic Model Registry
│   │   ├── base.py               # Abstract BaseIDSModel (.fit, .predict, .evaluate)
│   │   ├── tree_models.py        # [EXCLUSIVE: 4kub0]
│   │   └── boosting_models.py    # [EXCLUSIVE: lemonkartikeya]
│   └── evaluation/
│       ├── cascade.py            # Hierarchical L1 -> L2 triage runner
│       ├── evaluator.py          # Benchmark runner (iterates registry)
│       └── reporter.py           # Comparison tables & plots
├── tests/
│   ├── test_tree_models.py       # [EXCLUSIVE: 4kub0]
│   └── test_boosting_models.py   # [EXCLUSIVE: lemonkartikeya]
└── notebooks/
    ├── dev_4kub0/                # 4kub0's personal scratchpad
    └── kar/                      # lemonkartikeya's personal scratchpad (test.ipynb)
```

### Git Auto-Merge Rules:
1. **Contract-Driven**: Both agents implement `BaseIDSModel` (`.fit(X, y)`, `.predict(X)`, `.evaluate(X, y)`).
2. **File Isolation**: `4kub0`'s agent touches `src/models/tree_models.py`; `lemonkartikeya`'s agent touches `src/models/boosting_models.py`. Git merges them automatically without conflicts.
3. **Registry Pattern**: `src/models/__init__.py` registers all models dynamically. The benchmark runner loads all models without either agent modifying the runner script.
4. **Isolated Notebooks**: Scratchpads live in personal directories (`notebooks/dev_4kub0/` and `kar/`). Git never attempts to merge conflicting notebook JSON.

---

## 5. Part 2: Project Architecture (Simulation & Deployment) — *3rd Collaborator*

*Maintained for system integration context:*
- **2.1 Topology Design & iFogSim Setup**: IoT Traffic Generators $\rightarrow$ Edge Nodes $\rightarrow$ Fog Nodes $\rightarrow$ Cloud Data Center.
- **2.2 Tiered ML Model Placement**: Deploy lightweight Level 1 models on Edge nodes; deploy heavy Level 2 models on Cloud/Fog nodes.
- **2.3 Fault-Tolerance & Failover**: Heartbeat monitoring, dynamic rerouting if an edge node fails or suffers DDoS.
- **2.4 Simulation Profiling**: Track end-to-end latency, energy consumption, and recovery times under simulated failure.

---

## 6. Implementation Roadmap

1. **Step 1 (Contracts & Setup)**: Establish `src/contracts.py` and `src/models/base.py`. Ensure preprocessed data artifacts (`data/processed/`) and feature subsets (`feature_subsets.pkl`) are in place.
2. **Step 2 (Concurrent Model Training)**:
   - `4kub0` builds and tests `src/models/tree_models.py`.
   - `lemonkartikeya` builds and tests `src/models/boosting_models.py`.
3. **Step 3 (Checkpoint Review)**: Verify unit tests (`tests/test_tree_models.py` and `tests/test_boosting_models.py`). Compare latency vs. Recall trade-offs.
4. **Step 4 (Cascade Evaluation & Benchmarking)**: Run the full 36-experiment matrix across both levels and output `results/comparison_results.csv` and Pareto plots.
5. **Step 5 (Handoff to Part 2)**: Export top-performing Level 1 and Level 2 models (ONNX / PMML / joblib) to Collaborator 3 for iFogSim simulation.
