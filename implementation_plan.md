# Hierarchical IDS with Fog-Cloud Simulation: Project Pipeline & Architecture

This document elaborates on the proposed project, dividing it into two main phases: **Project Architecture (Simulation & Deployment)** and **Modelling (ML Pipeline & Evaluation)**.

## User Review Required

Please review the proposed pipeline and architecture below. Once approved, we can proceed with setting up the project structure and beginning the implementation phase.

## Open Questions

> [!IMPORTANT]
> - **Dataset Selection**: You mentioned 6 datasets but the validation matrix mentions 3 primary datasets. Which 3 datasets should we prioritize first? (e.g., IoT-23, Edge-IIoTset, CIC-IDS2017).
> - **Fog-Cloud Simulation**: Will we be integrating the trained ML models directly into iFogSim (Java), or simulating the network topology in iFogSim and evaluating the models in Python separately based on the simulated metrics? Calling Python models from Java can introduce overhead.
> - **Evolopy-FS Integration**: Evolopy-FS is a Python library. We will need to set it up and ensure compatibility with our datasets. Do you have a preference for which 6 evolutionary algorithms to use from the suite?

---

## Part 1: Modelling (ML Pipeline & Evaluation)

This phase focuses on developing the intelligence of the system using Python, scikit-learn, and Evolopy-FS.

### 1.1 Data Preprocessing & Balancing Module
- **Data Ingestion**: Scripts to load and parse the raw datasets (IoT-23, Edge-IIoTset, etc.).
- **Cleaning**: Imputation of missing values, removal of infinite values, and deduplication.
- **Encoding**: Label encoding or one-hot encoding for categorical features (e.g., protocols, states).
- **Scaling**: Min-Max scaling or Standard Scaling for numerical features.
- **Imbalance Handling**:
  - **Level 1 (Binary)**: SMOTE or Random Under-Sampling to balance Normal vs. Attack.
  - **Level 2 (Multiclass)**: Advanced SMOTE variants (e.g., Borderline-SMOTE, ADASYN) or class-weighted algorithms to handle $\le 1\%$ rare attack classes.

### 1.2 Evolutionary Feature Selection (Evolopy-FS)
- **Algorithm Setup**: Configure 6 metaheuristic algorithms (e.g., Genetic Algorithm (GA), Particle Swarm Optimization (PSO), Grey Wolf Optimizer (GWO), Ant Lion Optimizer (ALO), Whale Optimization Algorithm (WOA), Moth-Flame Optimization (MFO)).
- **Optimization Strategy**: Use a wrapper method with a fast classifier (like Decision Tree or Random Forest) to evaluate feature subset fitness.
- **Outputs**: Generate 6 distinct feature subsets per dataset (3 optimized for Binary, 3 optimized for Multiclass).

### 1.3 Hierarchical ML Engine Development
- **Level 1 (Binary Filter)**: Train models to distinguish Normal vs. Attack using the 3 Binary feature subsets. Focus on high recall (catch all attacks) and low latency.
- **Level 2 (Multiclass Categorizer)**: Train models to classify specific attack types using the 3 Multiclass feature subsets. Focus on high precision and F1-score across all classes.
- **Model Suite**: Implement pipelines for Random Forest (RF), Decision Trees/Extra Trees (DT/ET), LightGBM (LGBM), XGBoost (XGB), Gradient Boosting (GBM), and CatBoost.

### 1.4 Experimental Validation & Evaluation
- **Execution Matrix**: Run the 108 predefined experiments ($3 \text{ datasets} \times 6 \text{ feature sets} \times 6 \text{ algorithms}$).
- **Evaluation Metrics**: Calculate Accuracy, Precision, Recall, F1-score, False Positive Rate (FPR), and Inference Latency for each run.
- **Result Aggregation**: Generate comparative tables and visualizations (e.g., ROC curves, latency vs. accuracy plots) to identify the optimal Level 1 and Level 2 models.

---

## Part 2: Project Architecture (Simulation & Deployment)

This phase focuses on simulating the deployment environment and fault-tolerant mechanisms using iFogSim (Java).

### 2.1 Topology Design & iFogSim Setup
- **Environment**: Setup iFogSim using Maven/Gradle.
- **Network Topology Definition**:
  - **IoT Devices**: Traffic generators mimicking the datasets.
  - **Edge Nodes**: Gateways and routers placed close to the IoT devices.
  - **Fog Nodes**: Intermediate servers with moderate compute power.
  - **Cloud Data Center**: Centralized high-compute server.

### 2.2 Tiered ML Model Placement
- **Edge Layer Integration**: Deploy the lightweight Level 1 (Binary Filter) models on Edge nodes for near-zero latency triage.
- **Fog Layer Integration**: Implement intermediate aggregation logic and optionally deploy lighter versions of Level 2 models for regional analysis.
- **Cloud Layer Integration**: Deploy the heavy Level 2 (Multiclass) models for deep dive analysis, global threat intelligence, and model retraining.
- *Note: Integration might involve exporting Python models (e.g., using ONNX or PMML) and loading them in Java, or simulating the delay/compute cost based on Python results.*

### 2.3 Fault-Tolerance & Failover Mechanism
- **Health Monitoring**: Implement heartbeat mechanisms between Edge, Fog, and Cloud nodes.
- **Failover Logic**: If an Edge node fails or is compromised, traffic is automatically rerouted to a neighboring Edge node or directly to a Fog node.
- **Load Balancing**: Distribute processing load if a specific node experiences a DDoS attack.

### 2.4 Simulation Execution & System Profiling
- **Metrics Collection**: Track end-to-end latency, network usage, energy consumption, and Edge/Fog node resource utilization during simulation.
- **Resilience Testing**: Inject node failures during the simulation and measure system recovery time and attack detection rate degradation.

## Proposed Next Steps
1. Finalize the 3 primary datasets.
2. Initialize the Python environment and begin building the **Data Preprocessing & Balancing Module (1.1)**.
3. Simultaneously, setup the Java/iFogSim project structure for the **Architecture (Part 2)**.
