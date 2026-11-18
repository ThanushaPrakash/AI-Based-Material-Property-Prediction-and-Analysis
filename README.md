# ⚛️ AI-Based Material Property Prediction and Analysis
### Compound Explorer: An Explainable Machine Learning System for Materials Property Prediction and Recommendation

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Matbench Benchmark](https://img.shields.io/badge/Dataset-Matbench-green.svg)](https://matbench.materialsproject.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

**AI-Based Material Property Prediction and Analysis ("Compound Explorer")** is an explainable machine learning system designed to perform rapid preliminary analysis of inorganic materials directly from their chemical formulas. 

Given a chemical composition string (e.g., `TiO2`, `GaAs`, `BaTiO3`, `Fe2O3`, `CsPbI3`), the system:
1. Extracts **71+ composition-based material descriptors** using physical and elemental property statistics.
2. Predicts the **Experimental Band Gap** (in eV) using optimized ensemble and kernel regression models.
3. Classifies the material as **Metallic** or **Non-Metallic** with probability confidence scores.
4. Retrieves **Top-K Similar Materials** from Matbench benchmark datasets using feature-space cosine similarity.
5. Provides **Local & Global Feature Importance Interpretability** with chemical domain rationales.

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Ensure Python 3.10+ is installed on your system.

```bash
# Clone or navigate to project directory
cd "c:\Users\Meera\OneDrive\Documents\SEM 5\DDMM"

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run the End-to-End ML Pipeline (Data Extraction, Training & Tuning)
To download Matbench datasets, extract composition descriptors, train and optimize all ML models via 5-fold cross-validation, and build the similarity index:

```bash
python run_pipeline.py
```

### 3. Launch the Interactive Web Dashboard
Launch the Streamlit web application:

```bash
streamlit run app.py
```

The web dashboard will automatically open in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
├── app.py                      # Main Streamlit Web Application Dashboard
├── run_pipeline.py             # Full pipeline automation script (Data -> Features -> GridSearch CV -> Saved Models)
├── requirements.txt            # Python package dependencies
├── DOCUMENTATION.md            # Comprehensive Academic & Technical Project Report
├── README.md                   # Repository overview and setup guide
├── src/                        # Modular Source Code Package
│   ├── __init__.py
│   ├── data_loader.py          # Matbench dataset downloader & local cache manager
│   ├── feature_extractor.py     # Pymatgen/Matminer composition feature extractor (71 descriptors)
│   ├── train_models.py         # Regression & Classification model trainers + GridSearchCV 5-Fold CV
│   ├── similarity.py           # Nearest Neighbors similarity retriever engine
│   └── explainability.py       # Global Gini importance, local contributions & chemical rationale
└── models/                     # Saved Model Artifacts & Benchmark Results (Generated after pipeline run)
    ├── best_regression_model.pkl
    ├── best_classification_model.pkl
    ├── regression_scaler.pkl
    ├── classification_scaler.pkl
    ├── similarity_nn_model.pkl
    ├── regression_metrics.json
    └── classification_metrics.json
```

---

## 🧪 Tech Stack

- **Language**: Python 3.11
- **Materials Feature Engineering**: Pymatgen, Matminer
- **Machine Learning**: Scikit-Learn (Kernel Ridge Regression, Random Forest, Gradient Boosting, Extra Trees, Logistic Regression)
- **Optimization**: GridSearchCV with 5-Fold KFold & StratifiedKFold Cross-Validation
- **Similarity Search**: Scikit-Learn NearestNeighbors (Cosine & Euclidean metrics)
- **Visualization**: Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib
- **Web Interface**: Streamlit (Custom Dark/Glassmorphism Theme)

---

## 📊 Evaluation & Results

### Band Gap Regression Models (`matbench_expt_gap`)
- **Kernel Ridge Regression**: Radial basis function kernel tuned for regularization.
- **Random Forest Regressor**: 200 estimators with depth optimization.
- **Gradient Boosting Regressor**: Gradient boosted trees minimizing squared residual error.
- **Extra Trees Regressor**: Extremely randomized ensemble.

### Metallicity Classification Models (`matbench_expt_is_metal`)
- **Random Forest Classifier**: Classifies metallic vs. non-metallic state.
- **Gradient Boosting Classifier**: Sequential gradient boosting.
- **Extra Trees Classifier**: High-dimensional decision boundaries.
- **Logistic Regression**: Linear baseline with L2 penalty.

---

## 📄 Documentation

For detailed mathematical formulations, dataset statistics, feature engineering definitions, and full hyperparameter tuning tables, refer to [DOCUMENTATION.md](file:///c:/Users/Meera/OneDrive/Documents/SEM%205/DDMM/DOCUMENTATION.md).
