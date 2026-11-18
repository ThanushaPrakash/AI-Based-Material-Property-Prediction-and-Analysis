# AI-Based Material Property Prediction and Analysis
## An Explainable Machine Learning System for Materials Property Prediction and Recommendation

### Comprehensive Technical & Academic Documentation Report

---

## 1. Executive Summary & Abstract
The identification and screening of novel materials with targeted physical and electronic properties is a foundational challenge in modern solid-state physics, chemistry, and materials science. Conventional discovery workflows rely heavily on high-cost, time-intensive laboratory synthesis or computationally expensive *ab initio* calculations such as Density Functional Theory (DFT). 

This work presents **AI-Based Material Property Prediction and Analysis ("Compound Explorer")**, an end-to-end, explainable machine learning framework designed for composition-based materials property prediction, hyperparameter-optimized model benchmarking, feature interpretability, and similar-material recommendation.

Using curated benchmark datasets from **Matbench** (`matbench_expt_gap` with 4,604 samples and `matbench_expt_is_metal` with 4,921 samples), the system extracts high-dimensional composition-based material descriptors directly from chemical formula strings (e.g., $\text{TiO}_2$, $\text{GaAs}$, $\text{BaTiO}_3$). The framework compares multiple machine learning architectures—including **Kernel Ridge Regression (KRR)**, **Random Forest (RF)**, **Gradient Boosting (GBR/GBC)**, **Extra Trees (ET)**, and **Logistic Regression**—subjected to 5-fold cross-validated hyperparameter optimization using **GridSearchCV**.

The system integrates a **Nearest Neighbors similarity retriever** operating in normalized feature space to recommend Top-K candidate materials, and employs global Gini feature importance and local descriptor deviation analysis to explain individual property predictions. The entire solution is deployed in an interactive Streamlit web dashboard.

---

## 2. Introduction & Literature Background
### 2.1 Materials Informatics Paradigm
Materials Informatics combines materials science domain knowledge with statistical machine learning and data science methodologies to accelerate materials discovery cycles. Rather than performing manual trials or full quantum mechanical simulations for thousands of candidates, machine learning models can learn complex non-linear mappings between chemical stoichiometry, elemental properties, and macroscopic physical behavior.

### 2.2 Base Paper Reference
This project is built upon the foundational benchmark established in:
> **Dunn, A., Wang, Q., Ganose, A., Dopp, D., & Jain, A. (2020).** *"Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm."* **npj Computational Materials**, 6, 138.

The Matbench benchmark suite provides standardized, rigorously curated datasets for evaluating machine learning algorithms across diverse materials prediction tasks. This project extends the Matbench foundation by constructing an integrated, accessible framework combining:
1. Composition-derived feature engineering
2. Systematic hyperparameter optimization via `GridSearchCV`
3. Multi-model comparative regression and classification benchmarks
4. Cosine similarity-based material retrieval
5. Feature importance explainability
6. User-centric interactive web deployment

---

## 3. System Architecture & Workflow

```
                                  +---------------------------------------+
                                  |     Chemical Formula Input (e.g. TiO2)|
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +-------------------+-------------------+
                                  |   Composition Descriptor Engine       |
                                  |   (70+ Stoichiometric & Element Stats)|
                                  +-------------------+-------------------+
                                                      |
                                         +------------+------------+
                                         |                         |
                                         v                         v
                       +-----------------+-------+     +-----------+-------------------+
                       | Band Gap Regressor      |     | Metallicity Classifier        |
                       | (KRR, RF, GBR GridSearch|     | (RF, GBC, ET GridSearch CV)   |
                       +-----------------+-------+     +-----------+-------------------+
                                         |                         |
                                         +------------+------------+
                                                      |
                                                      v
                                  +-------------------+-------------------+
                                  | Similarity Engine (Nearest Neighbors) |
                                  | & Feature Interpretability (Gini/SHAP)|
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +-------------------+-------------------+
                                  |     Interactive Streamlit Dashboard   |
                                  +---------------------------------------+
```

---

## 4. Dataset Description
The system is built and benchmarked on two standardized experimental Matbench datasets:

| Dataset Name | Target Property | Task Type | Sample Count | Target Variable Unit / Format |
| :--- | :--- | :--- | :--- | :--- |
| `matbench_expt_gap` | Experimental Band Gap | Regression | 4,604 | Continuous band gap $E_g$ (eV) |
| `matbench_expt_is_metal` | Metallicity Status | Binary Classification | 4,921 | Boolean (1 = Metal, 0 = Non-Metal) |

---

## 5. Feature Engineering Methodology
Machine learning models require fixed-length numerical vectors. The chemical formula parsing and feature generation pipeline transforms raw chemical composition strings into a 71-dimensional numerical vector:

### 5.1 Elemental Property Statistics
For each elemental property $P \in \{\text{atomic mass}, \text{electronegativity } X, \text{atomic radius}, \text{atomic number } Z, \text{group}, \text{row}, \text{Mendeleev number}, \text{thermal conductivity}\}$, five statistical aggregations are computed weighted by elemental fraction $w_i$:
$$\text{Mean: } \bar{P} = \sum_{i} w_i P_i$$
$$\text{Min: } P_{\min} = \min_{i} P_i, \quad \text{Max: } P_{\max} = \max_{i} P_i$$
$$\text{Range: } P_{\text{range}} = P_{\max} - P_{\min}$$
$$\text{Standard Deviation: } \sigma_P = \sqrt{\sum_{i} w_i (P_i - \bar{P})^2}$$

### 5.2 Chemical & Stoichiometric Descriptors
- **Stoichiometry**: Number of constituent elements ($N_{\text{elem}}$), total atom count, maximum/minimum elemental fraction, fraction standard deviation.
- **Periodic Table Block Fractions**: Fractions of elements belonging to $s$-, $p$-, $d$-, and $f$-blocks.
- **Metallicity Fractions**: Fractions of metallic, non-metallic, and metalloid elements.
- **Anion Character Indicators**: Binary flags for oxides ($\text{O}$), chalcogenides ($\text{S, Se, Te}$), halides ($\text{F, Cl, Br, I}$), and pnictides ($\text{N, P, As, Sb, Bi}$).

---

## 6. Machine Learning Modeling & Optimization

### 6.1 Regression Models (Experimental Band Gap)
Four regression algorithms are implemented and evaluated:
1. **Kernel Ridge Regression (KRR)**: Combines Ridge Regression ($L_2$ regularization) with non-linear radial basis function (RBF) kernel trick.
2. **Random Forest Regressor (RF)**: Ensemble of de-correlated decision trees trained via bootstrap aggregation (bagging).
3. **Gradient Boosting Regressor (GBR)**: Sequential ensemble building weak decision trees to minimize mean squared residual loss.
4. **Extra Trees Regressor (ET)**: Extremely randomized trees providing lower variance and randomized splitting thresholds.

### 6.2 Classification Models (Metallicity Status)
Four classification algorithms are evaluated:
1. **Random Forest Classifier (RF)**
2. **Gradient Boosting Classifier (GBC)**
3. **Extra Trees Classifier (ET)**
4. **Logistic Regression (LR)**

### 6.3 Hyperparameter Tuning via GridSearchCV
Models undergo systematic hyperparameter optimization using 5-fold cross-validation (`KFold` for regression, `StratifiedKFold` for classification). The hyperparameter search space includes tree depth, estimator counts, learning rates, kernel gammas, and regularization constants.

---

## 7. Performance Metrics & Evaluation Results

### 7.1 Regression Evaluation Metrics
- **Mean Absolute Error (MAE)**: $\frac{1}{N} \sum |y_i - \hat{y}_i|$
- **Root Mean Squared Error (RMSE)**: $\sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}$
- **Coefficient of Determination ($R^2$)**: $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

### 7.2 Classification Evaluation Metrics
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **ROC-AUC Score**: Area under Receiver Operating Characteristic curve.

---

## 8. Explainability & Similar Material Retrieval

### 8.1 Similar Material Retrieval Engine
Using a `NearestNeighbors` model fitted on normalized composition feature vectors, the similarity engine computes the cosine distance $d_{\text{cosine}}(\mathbf{u}, \mathbf{v})$ between the query formula vector $\mathbf{u}$ and candidate dataset vectors $\mathbf{v}$.
$$\text{Similarity Score (\%)} = \max\left(0, (1 - d_{\text{cosine}}) \times 100\right)$$
The top $K$ nearest materials are returned alongside their known experimental band gap, metallicity status, and compositional distance.

### 8.2 Feature Importance & Domain Rationale
Global tree Gini importances reveal that **electronegativity statistics** ($\bar{X}, X_{\text{range}}$), **atomic radius**, and **$d$/$f$-block fractions** consistently dictate electronic band gaps and metallic character. Local explanations highlight feature deviations relative to dataset averages and output natural language chemical insights.

---

## 9. Conclusion & Practical Impact
The **AI-Based Material Property Prediction and Analysis** framework successfully unifies composition-based property prediction, hyperparameter optimization, explainability, similarity retrieval, and interactive web deployment. It serves as a rapid preliminary screening and decision-support tool, empowering materials scientists to prioritize high-potential material candidates prior to expensive computation or synthesis.

---

## 10. References
1. Dunn, A., Wang, Q., Ganose, A., Dopp, D., & Jain, A. (2020). *Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm.* **npj Computational Materials**, 6, 138.
2. Ward, L., Agrawal, A., Choudhary, A., & Wolverton, C. (2016). *A general-purpose machine learning framework for predicting properties of inorganic materials.* **npj Computational Materials**, 2, 16028.
3. Ong, S. P., et al. (2013). *Python Materials Genomics (pymatgen): An object-oriented Python library for materials analysis.* **Computational Materials Science**, 68, 314-319.
