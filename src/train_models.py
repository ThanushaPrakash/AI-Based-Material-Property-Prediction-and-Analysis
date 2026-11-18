"""
Model Training, Optimization, and Evaluation Module.
Implements Kernel Ridge Regression, Random Forest, Gradient Boosting, Extra Trees,
GridSearchCV hyperparameter optimization, and comprehensive evaluation metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
)

from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def train_and_optimize_regression(X, y):
    """
    Train and tune regression models for Experimental Band Gap prediction.
    Models compared: Kernel Ridge, Random Forest, Gradient Boosting, Extra Trees.
    """
    print("\n=======================================================")
    print("      TRAINING & TUNING BAND GAP REGRESSION MODELS     ")
    print("=======================================================")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Base Models (Default Hyperparameters)
    base_models = {
        'Kernel Ridge': KernelRidge(alpha=1.0, kernel='rbf'),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }

    base_results = {}
    for name, model in base_models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        base_results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"[Base] {name:20s} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

    # 2. Hyperparameter Optimization via GridSearchCV
    param_grids = {
        'Kernel Ridge': {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [0.01, 0.1, 1.0]
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [15, 25, None],
            'min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        },
        'Extra Trees': {
            'n_estimators': [100, 200],
            'max_depth': [15, 25, None]
        }
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    tuned_models = {}
    tuned_results = {}
    best_params = {}
    test_predictions = {}

    best_overall_r2 = -float('inf')
    best_overall_name = None
    best_overall_model = None

    for name, base_model in base_models.items():
        print(f"\nGridSearchCV for {name}...")
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grids[name],
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_train_scaled, y_train)
        best_m = grid.best_estimator_
        y_pred = best_m.predict(X_test_scaled)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        tuned_models[name] = best_m
        tuned_results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        best_params[name] = grid.best_params_
        test_predictions[name] = {'y_true': y_test.tolist(), 'y_pred': y_pred.tolist()}

        print(f"[Tuned] {name:19s} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
        print(f"        Best Params: {grid.best_params_}")

        if r2 > best_overall_r2:
            best_overall_r2 = r2
            best_overall_name = name
            best_overall_model = best_m

    # Save artifacts
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'regression_scaler.pkl'))
    joblib.dump(best_overall_model, os.path.join(MODEL_DIR, 'best_regression_model.pkl'))
    joblib.dump(tuned_models, os.path.join(MODEL_DIR, 'all_regression_models.pkl'))

    reg_summary = {
        'base_results': base_results,
        'tuned_results': tuned_results,
        'best_params': best_params,
        'best_model_name': best_overall_name,
        'test_predictions': test_predictions
    }

    with open(os.path.join(MODEL_DIR, 'regression_metrics.json'), 'w') as f:
        json.dump(reg_summary, f, indent=4)

    return reg_summary, scaler, best_overall_model


def train_and_optimize_classification(X, y):
    """
    Train and tune classification models for Metallicity prediction.
    Models compared: Random Forest, Gradient Boosting, Extra Trees, Logistic Regression.
    """
    print("\n=======================================================")
    print("      TRAINING & TUNING METALLICITY CLASSIFIERS        ")
    print("=======================================================")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Base Classifiers
    base_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    base_results = {}
    for name, model in base_models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        base_results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
        print(f"[Base] {name:20s} -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

    # 2. Hyperparameter Optimization
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [15, 25, None],
            'min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        },
        'Extra Trees': {
            'n_estimators': [100, 200],
            'max_depth': [15, 25, None]
        },
        'Logistic Regression': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2']
        }
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tuned_models = {}
    tuned_results = {}
    best_params = {}
    confusion_matrices = {}
    roc_curves = {}

    best_overall_f1 = -1.0
    best_overall_name = None
    best_overall_model = None

    for name, base_model in base_models.items():
        print(f"\nGridSearchCV for {name}...")
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grids[name],
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_train_scaled, y_train)
        best_m = grid.best_estimator_
        y_pred = best_m.predict(X_test_scaled)
        y_proba = best_m.predict_proba(X_test_scaled)[:, 1] if hasattr(best_m, 'predict_proba') else None

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else 0.0

        cm = confusion_matrix(y_test, y_pred).tolist()

        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_curves[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': auc}

        tuned_models[name] = best_m
        tuned_results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'ROC_AUC': auc}
        best_params[name] = grid.best_params_
        confusion_matrices[name] = cm

        print(f"[Tuned] {name:19s} -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print(f"        Best Params: {grid.best_params_}")

        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall_name = name
            best_overall_model = best_m

    # Save artifacts
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'classification_scaler.pkl'))
    joblib.dump(best_overall_model, os.path.join(MODEL_DIR, 'best_classification_model.pkl'))
    joblib.dump(tuned_models, os.path.join(MODEL_DIR, 'all_classification_models.pkl'))

    clf_summary = {
        'base_results': base_results,
        'tuned_results': tuned_results,
        'best_params': best_params,
        'best_model_name': best_overall_name,
        'confusion_matrices': confusion_matrices,
        'roc_curves': roc_curves
    }

    with open(os.path.join(MODEL_DIR, 'classification_metrics.json'), 'w') as f:
        json.dump(clf_summary, f, indent=4)

    return clf_summary, scaler, best_overall_model
