"""
Composition Feature Extractor Module.
Generates composition-based material descriptors directly from chemical formulas
using Pymatgen element & stoichiometry statistics.
"""

import os
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element


class CompositionFeatureExtractor:
    """Extracts numerical composition descriptors from chemical formula strings."""

    PROPERTIES = [
        'atomic_mass',
        'X',  # Pauling Electronegativity
        'atomic_radius',
        'Z',  # Atomic Number
        'group',
        'row',
        'mendeleev_no',
        'thermal_conductivity',
        'electrical_resistivity',
        'valence',
        'average_ionic_radius',
    ]

    BLOCK_ELEMENTS = {
        's': ['H', 'Li', 'Be', 'Na', 'Mg', 'K', 'Ca', 'Rb', 'Sr', 'Cs', 'Ba', 'Fr', 'Ra'],
        'p': ['B', 'C', 'N', 'O', 'F', 'Ne', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'],
        'd': ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn'],
        'f': ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
    }

    HALIDES = {'F', 'Cl', 'Br', 'I', 'At'}
    CHALCOGENIDES = {'S', 'Se', 'Te'}
    PNICTIDES = {'N', 'P', 'As', 'Sb', 'Bi'}

    def __init__(self):
        self.feature_names = []

    def featurize_formula(self, formula_str):
        """Extract a dict of composition features for a single formula string."""
        comp = Composition(formula_str)
        el_amt = comp.get_el_amt_dict()
        total_atoms = float(sum(el_amt.values()))
        fractions = {el: amt / total_atoms for el, amt in el_amt.items()}

        feats = {}

        # Stoichiometry features
        feats['n_elements'] = len(el_amt)
        feats['total_atoms'] = total_atoms
        frac_vals = list(fractions.values())
        feats['max_fraction'] = max(frac_vals)
        feats['min_fraction'] = min(frac_vals)
        feats['fraction_std'] = float(np.std(frac_vals)) if len(frac_vals) > 1 else 0.0

        # Block & chemical class fractions
        s_frac, p_frac, d_frac, f_frac = 0.0, 0.0, 0.0, 0.0
        metal_frac, nonmetal_frac, metalloid_frac = 0.0, 0.0, 0.0

        for el_sym, frac in fractions.items():
            el = Element(el_sym)
            if el_sym in self.BLOCK_ELEMENTS['s']:
                s_frac += frac
            elif el_sym in self.BLOCK_ELEMENTS['p']:
                p_frac += frac
            elif el_sym in self.BLOCK_ELEMENTS['d']:
                d_frac += frac
            elif el_sym in self.BLOCK_ELEMENTS['f']:
                f_frac += frac

            if el.is_metal:
                metal_frac += frac
            elif el.is_metalloid:
                metalloid_frac += frac
            else:
                nonmetal_frac += frac

        feats['fraction_s_block'] = s_frac
        feats['fraction_p_block'] = p_frac
        feats['fraction_d_block'] = d_frac
        feats['fraction_f_block'] = f_frac
        feats['fraction_metal'] = metal_frac
        feats['fraction_nonmetal'] = nonmetal_frac
        feats['fraction_metalloid'] = metalloid_frac

        # Specific anion indicators
        feats['is_oxide'] = 1.0 if 'O' in el_amt else 0.0
        feats['is_chalcogenide'] = 1.0 if any(el in self.CHALCOGENIDES for el in el_amt) else 0.0
        feats['is_halide'] = 1.0 if any(el in self.HALIDES for el in el_amt) else 0.0
        feats['is_pnictide'] = 1.0 if any(el in self.PNICTIDES for el in el_amt) else 0.0

        # Elemental property statistics
        for prop in self.PROPERTIES:
            vals = []
            weights = []
            for el_sym, frac in fractions.items():
                el = Element(el_sym)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        val = getattr(el, prop, None)
                    except Exception:
                        val = 0.0
                if val is None or (isinstance(val, (float, int)) and (np.isnan(val) or np.isinf(val))):
                    val = 0.0
                elif not isinstance(val, (float, int)):
                    try:
                        val = float(val)
                    except Exception:
                        val = 0.0
                vals.append(float(val))
                weights.append(frac)

            vals = np.array(vals)
            weights = np.array(weights)

            w_mean = float(np.sum(vals * weights))
            w_var = float(np.sum(weights * (vals - w_mean) ** 2))
            w_std = float(np.sqrt(w_var))

            feats[f'{prop}_mean'] = w_mean
            feats[f'{prop}_min'] = float(np.min(vals))
            feats[f'{prop}_max'] = float(np.max(vals))
            feats[f'{prop}_range'] = float(np.max(vals) - np.min(vals))
            feats[f'{prop}_std'] = w_std

        return feats

    def featurize_dataframe(self, df, formula_col='formula'):
        """Featurize a DataFrame containing a column of chemical formulas."""
        feature_dicts = []
        valid_indices = []

        for idx, formula in enumerate(df[formula_col]):
            try:
                fd = self.featurize_formula(formula)
                feature_dicts.append(fd)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Skipping invalid formula {formula}: {e}")

        feat_df = pd.DataFrame(feature_dicts)
        # Fill any unexpected missing values with median
        feat_df = feat_df.fillna(feat_df.median(numeric_only=True)).fillna(0.0)

        if not self.feature_names:
            self.feature_names = list(feat_df.columns)

        return feat_df, valid_indices


if __name__ == '__main__':
    fe = CompositionFeatureExtractor()
    sample_formulas = ['TiO2', 'GaAs', 'Fe2O3', 'BaTiO3', 'NaCl', 'CsPbI3']
    df_sample = pd.DataFrame({'formula': sample_formulas})
    feats, _ = fe.featurize_dataframe(df_sample)
    print("Features shape:", feats.shape)
    print("Feature columns:", feats.columns.tolist()[:15])
