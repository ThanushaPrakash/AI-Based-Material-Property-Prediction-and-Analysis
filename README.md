# ⚛️ Compound Explorer

### An Explainable Machine Learning System for Materials Property Prediction and Recommendation

\<p align="center">   \<img src="[https://img.shields.io/badge/Python-3.11-blue?logo=python](https://img.shields.io/badge/Python-3.11-blue?logo=python)" />   \<img src="[https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit)" />   \<img src="[https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?logo=scikitlearn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?logo=scikitlearn)" />   \<img src="[https://img.shields.io/badge/Dataset-Matbench-green](https://img.shields.io/badge/Dataset-Matbench-green)" /> \</p>

---

## 🌌 Overview

**Compound Explorer** is an explainable machine learning application designed to predict material properties, analyze chemical compositions, and recommend similar materials using machine learning techniques.

The system accepts chemical formulas such as `TiO2`, `GaAs`, `BaTiO3`, and `Fe2O3`, extracts composition-based material descriptors, and performs two major prediction tasks:

- **Band Gap Prediction:** Estimates experimental band gap values using regression models.
- **Metallicity Classification:** Classifies materials as metallic or non-metallic with probability confidence scores.

Beyond prediction, Compound Explorer integrates similar-material retrieval and feature importance analysis to provide insights into the properties influencing individual predictions.

The application is deployed through an interactive Streamlit dashboard.

---

## 🎯 Key Features

- **Chemical Formula Processing:** Converts chemical compositions into numerical material descriptors.
- **71-Dimensional Feature Extraction:** Generates composition-based features using elemental and stoichiometric properties.
- **Band Gap Prediction:** Predicts experimental band gap values in electron volts (eV).
- **Metallicity Classification:** Identifies materials as metallic or non-metallic.
- **Model Benchmarking:** Compares multiple machine learning algorithms.
- **Hyperparameter Optimization:** Uses GridSearchCV with 5-fold cross-validation.
- **Similar Material Recommendation:** Retrieves top-K similar materials using nearest-neighbor similarity.
- **Explainable AI:** Provides global feature importance and local prediction explanations.
- **Interactive Dashboard:** Presents predictions, model comparisons, visualizations, and material insights.

---

## 🧪 Machine Learning Tasks

### 1. Experimental Band Gap Prediction

A regression task that estimates the experimental electronic band gap of a material.

Dataset: `matbench_expt_gap`

Algorithms explored:

- Kernel Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor

Evaluation metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### 2. Metallicity Classification

A binary classification task that determines whether a material is metallic or non-metallic.

Dataset: `matbench_expt_is_metal`

Algorithms explored:

- Random Forest Classifier
- Gradient Boosting Classifier
- Extra Trees Classifier
- Logistic Regression

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## 📊 Dataset

The project uses experimental material property datasets from **Matbench**.

| Dataset                | Prediction Task            | Samples |
| ---------------------- | -------------------------- | ------: |
| matbench_expt_gap      | Band Gap Regression        |   4,604 |
| matbench_expt_is_metal | Metallicity Classification |   4,921 |

The datasets provide experimental material properties for training and evaluating the machine learning models.

---

## ⚙️ Feature Engineering

Chemical formulas are transformed into 71-dimensional numerical feature vectors.

The extracted descriptors include:

- Elemental property statistics
- Atomic mass
- Electronegativity
- Atomic radius
- Atomic number
- Elemental group and period
- Mendeleev number
- Thermal conductivity
- Stoichiometric and compositional descriptors

These features represent the chemical composition in a format suitable for machine learning algorithms.

---

## 🏗️ System Architecture

```text
       Chemical Formula Input
                 |
                 v
      Composition Feature Extraction
            (71 Descriptors)
                 |
        +--------+--------+
        |                 |
        v                 v
   Band Gap          Metallicity
   Regression        Classification
        |                 |
        +--------+--------+
                 |
                 v
       Prediction & Confidence
                 |
        +--------+--------+
        |                 |
        v                 v
 Similar Material    Explainability
    Retrieval          Analysis
        |                 |
        +--------+--------+
                 |
                 v
       Streamlit Dashboard
```

---

## 🧠 Explainable AI & Similarity Analysis

### Feature Importance

The system uses feature importance analysis to identify the material descriptors that contribute to model predictions.

It provides:

- Global feature importance for regression and classification models.
- Local descriptor deviation analysis.
- Chemical rationale summaries for individual predictions.

### Similar Material Recommendation

A Nearest Neighbors-based retrieval engine identifies similar materials using normalized composition feature vectors.

- Uses cosine similarity and feature-space distance.
- Retrieves Top-K similar materials.
- Displays associated experimental properties where available.

---

## 🛠️ Technology Stack

| Category              | Technologies                        |
| --------------------- | ----------------------------------- |
| Programming Language  | Python                              |
| Machine Learning      | Scikit-learn                        |
| Materials Informatics | Pymatgen, Matminer                  |
| Benchmark Datasets    | Matbench                            |
| Feature Engineering   | Composition-based descriptors       |
| Model Optimization    | GridSearchCV                        |
| Similarity Search     | NearestNeighbors                    |
| Explainability        | Feature Importance, SHAP dependency |
| Visualization         | Plotly, Matplotlib, Seaborn         |
| Web Application       | Streamlit                           |
| Data Processing       | Pandas, NumPy                       |

---

## 📂 Project Structure

```text
Compound-Explorer/
│
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
├── DOCUMENTATION.md
│
├── data/
│   ├── matbench_expt_gap.csv
│   └── matbench_expt_is_metal.csv
│
├── models/
│   ├── all_classification_models.pkl
│   ├── all_regression_models.pkl
│   ├── best_classification_model.pkl
│   ├── best_regression_model.pkl
│   ├── classification_metrics.json
│   ├── regression_metrics.json
│   ├── classification_scaler.pkl
│   ├── regression_scaler.pkl
│   └── similarity models and feature artifacts
│
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── feature_extractor.py
    ├── train_models.py
    ├── similarity.py
    └── explainability.py
```

---

## 🚀 Installation and Execution

### Prerequisites

- Python 3.10 or above
- pip
- Internet connection for downloading Matbench datasets during pipeline execution

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Compound-Explorer.git
```

### 2. Navigate to the Project Directory

```bash
cd Compound-Explorer
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```powershell
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Machine Learning Pipeline

```bash
python run_pipeline.py
```

This executes the data preparation, feature extraction, model training, hyperparameter optimization, evaluation, and similarity-index preparation workflow.

### 6. Launch the Streamlit Application

```bash
streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

---

## 📈 Model Evaluation

The application includes comparative evaluation of default and hyperparameter-optimized models.

**Regression:** MAE, RMSE, and R².

**Classification:** Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

The dashboard provides model comparison tables, performance visualizations, confusion matrices, ROC curves, and feature importance plots.

---

## 📚 Reference

Dunn, A., Wang, Q., Ganose, A., Dopp, D., & Jain, A. (2020).

*Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm.*

npj Computational Materials, 6, 138.

---

## 👩‍💻 Author

**Thanusha Prakash**

Computer Science and Engineering | Artificial Intelligence

[GitHub](https://github.com/ThanushaPrakash)

---

