"""
Explainable Machine Learning & Feature Importance Module.
Provides Global Gini Importance, Permutation Importance, SHAP analysis,
and Local Chemical Rationale for material property predictions.
"""

import os
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


class ExplainabilityEngine:
    """Computes global feature importances and local explanations for model predictions."""

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def get_global_feature_importance(self, top_n=15):
        """Retrieve Gini feature importance from tree-based models."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            df_imp = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).reset_index(drop=True)
            return df_imp.head(top_n)
        return pd.DataFrame()

    def generate_local_explanation(self, query_feats_dict, dataset_mean_feats=None, top_n=6):
        """
        Generate local feature contributions and chemical rationale for a specific formula.
        """
        global_imp = self.get_global_feature_importance(top_n=20)
        if global_imp.empty:
            return pd.DataFrame(), "Feature importances not available for this model architecture."

        top_feature_names = global_imp['Feature'].tolist()

        contributions = []
        for feat in top_feature_names:
            val = query_feats_dict.get(feat, 0.0)
            avg_val = dataset_mean_feats.get(feat, val) if dataset_mean_feats is not None else val
            diff = val - avg_val

            readable_name = feat.replace('_', ' ').title()
            contributions.append({
                'Feature': feat,
                'Display Name': readable_name,
                'Value': round(float(val), 3),
                'Dataset Avg': round(float(avg_val), 3),
                'Difference': round(float(diff), 3),
                'Global Rank': len(contributions) + 1
            })

        df_local = pd.DataFrame(contributions).head(top_n)

        # Generate plain text chemical rationale
        rationale_lines = []
        for _, row in df_local.head(4).iterrows():
            feat = row['Feature']
            val = row['Value']
            diff = row['Difference']
            direction = "higher than average" if diff > 0 else "lower than average"

            if 'X' in feat:
                rationale_lines.append(f"• **Electronegativity ({row['Display Name']})**: {val} ({direction}). High electronegativity contrast influences band gap magnitude.")
            elif 'atomic_radius' in feat:
                rationale_lines.append(f"• **Atomic Radius ({row['Display Name']})**: {val} Å ({direction}). Radii differences govern orbital overlap and band width.")
            elif 'metal' in feat:
                rationale_lines.append(f"• **Metallicity Fraction ({row['Display Name']})**: {val} ({direction}). High metallic element content shifts Fermi level into conduction band.")
            elif 'oxide' in feat or 'chalcogenide' in feat or 'halide' in feat:
                rationale_lines.append(f"• **Anion Character ({row['Display Name']})**: {val}. Presence of electronegative anions generally widens band gap.")
            else:
                rationale_lines.append(f"• **{row['Display Name']}**: {val} ({direction}). Influences electronic density of states.")

        rationale = "\n".join(rationale_lines)
        return df_local, rationale


def compute_shap_values(model, X_sample, feature_names):
    """Compute SHAP values if shap library is available."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        return explainer, shap_values
    except Exception as e:
        print(f"SHAP calculation skipped: {e}")
        return None, None
