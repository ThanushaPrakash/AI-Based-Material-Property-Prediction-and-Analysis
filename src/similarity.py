"""
Similar Material Retrieval & Recommendation Module.
Uses NearestNeighbors on composition-derived feature representations
to retrieve Top-K compositionally and property-similar candidate materials.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


class MaterialSimilarityEngine:
    """Finds top-K compositionally similar materials using feature space distance."""

    def __init__(self, metric='cosine'):
        self.metric = metric
        self.nn_model = None
        self.scaler = None
        self.dataset_df = None
        self.feature_matrix = None

    def fit(self, dataset_df, feature_matrix, feature_names):
        """Fit NearestNeighbors index on dataset features."""
        self.dataset_df = dataset_df.copy().reset_index(drop=True)
        self.feature_names = feature_names

        self.scaler = StandardScaler()
        self.feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)

        self.nn_model = NearestNeighbors(n_neighbors=20, metric=self.metric, algorithm='brute')
        self.nn_model.fit(self.feature_matrix_scaled)

        # Save artifacts
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, 'similarity_scaler.pkl'))
        joblib.dump(self.nn_model, os.path.join(MODEL_DIR, 'similarity_nn_model.pkl'))
        joblib.dump(self.dataset_df, os.path.join(MODEL_DIR, 'similarity_dataset.pkl'))
        joblib.dump(self.feature_matrix_scaled, os.path.join(MODEL_DIR, 'similarity_feature_matrix.pkl'))
        joblib.dump(self.feature_names, os.path.join(MODEL_DIR, 'similarity_feature_names.pkl'))

    def find_similar_materials(self, query_feature_dict, top_k=5):
        """Given a feature dictionary for a query formula, find top_k similar materials."""
        if self.nn_model is None or not self.feature_names:
            # Try to load cached artifacts
            self.load_artifacts()

        query_vec = pd.DataFrame([query_feature_dict])[self.feature_names].values
        query_vec_scaled = self.scaler.transform(query_vec)

        distances, indices = self.nn_model.kneighbors(query_vec_scaled, n_neighbors=top_k + 5)

        results = []
        seen_formulas = set()

        for dist, idx in zip(distances[0], indices[0]):
            row = self.dataset_df.iloc[idx]
            formula = row['formula']

            if formula in seen_formulas:
                continue
            seen_formulas.add(formula)

            # Similarity score: 0 to 100%
            if self.metric == 'cosine':
                similarity_pct = max(0.0, (1.0 - dist) * 100.0)
            else:
                similarity_pct = max(0.0, 100.0 / (1.0 + dist))

            res_entry = {
                'Formula': formula,
                'Similarity (%)': round(similarity_pct, 1),
                'Distance': round(float(dist), 4),
            }

            if 'gap_expt' in row:
                res_entry['Band Gap (eV)'] = round(float(row['gap_expt']), 2)
            if 'is_metal' in row:
                res_entry['Is Metal'] = "Metal" if bool(row['is_metal']) else "Non-Metal"

            results.append(res_entry)

            if len(results) >= top_k:
                break

        return pd.DataFrame(results)

    def load_artifacts(self):
        """Load fitted similarity engine from disk."""
        self.scaler = joblib.load(os.path.join(MODEL_DIR, 'similarity_scaler.pkl'))
        self.nn_model = joblib.load(os.path.join(MODEL_DIR, 'similarity_nn_model.pkl'))
        self.dataset_df = joblib.load(os.path.join(MODEL_DIR, 'similarity_dataset.pkl'))
        self.feature_matrix_scaled = joblib.load(os.path.join(MODEL_DIR, 'similarity_feature_matrix.pkl'))
        
        sim_fn_path = os.path.join(MODEL_DIR, 'similarity_feature_names.pkl')
        global_fn_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
        if os.path.exists(sim_fn_path):
            self.feature_names = joblib.load(sim_fn_path)
        elif os.path.exists(global_fn_path):
            self.feature_names = joblib.load(global_fn_path)
        elif hasattr(self.scaler, 'feature_names_in_'):
            self.feature_names = self.scaler.feature_names_in_.tolist()
        else:
            self.feature_names = []
