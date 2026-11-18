"""
Data Loader Module for Matbench Datasets.
Loads matbench_expt_gap and matbench_expt_is_metal datasets via matminer/matbench
with local caching support.
"""

import os
import pandas as pd
from pymatgen.core import Composition

# Apply pymatgen compatibility patch for matminer if needed
import pymatgen.core.periodic_table as pt
if not hasattr(pt, '_pt_data'):
    import json
    pt_data_path = os.path.join(os.path.dirname(pt.__file__), 'periodic_table.json')
    if os.path.exists(pt_data_path):
        with open(pt_data_path) as f:
            pt._pt_data = json.load(f)
    else:
        pt._pt_data = {}

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(CACHE_DIR, exist_ok=True)


def load_bandgap_dataset(force_download=False):
    """
    Load matbench_expt_gap dataset (Experimental Band Gap in eV).
    Returns pandas DataFrame with columns ['composition', 'gap_expt', 'formula']
    """
    cache_path = os.path.join(CACHE_DIR, 'matbench_expt_gap.csv')
    if os.path.exists(cache_path) and not force_download:
        df = pd.read_csv(cache_path)
        return df

    print("Fetching matbench_expt_gap dataset...")
    try:
        from matminer.datasets import load_dataset
        df = load_dataset('matbench_expt_gap')
        df = df.rename(columns={'gap expt': 'gap_expt'})
        df['formula'] = df['composition'].astype(str)
        # Filter valid compositions
        valid_mask = df['formula'].apply(is_valid_composition)
        df = df[valid_mask].reset_index(drop=True)
        df.to_csv(cache_path, index=False)
        return df
    except Exception as e:
        print(f"Error loading via matminer: {e}")
        raise e


def load_metallicity_dataset(force_download=False):
    """
    Load matbench_expt_is_metal dataset (Metallicity binary classification).
    Returns pandas DataFrame with columns ['composition', 'is_metal', 'formula']
    """
    cache_path = os.path.join(CACHE_DIR, 'matbench_expt_is_metal.csv')
    if os.path.exists(cache_path) and not force_download:
        df = pd.read_csv(cache_path)
        return df

    print("Fetching matbench_expt_is_metal dataset...")
    try:
        from matminer.datasets import load_dataset
        df = load_dataset('matbench_expt_is_metal')
        df['formula'] = df['composition'].astype(str)
        # Filter valid compositions
        valid_mask = df['formula'].apply(is_valid_composition)
        df = df[valid_mask].reset_index(drop=True)
        df['is_metal'] = df['is_metal'].astype(int)
        df.to_csv(cache_path, index=False)
        return df
    except Exception as e:
        print(f"Error loading via matminer: {e}")
        raise e


def is_valid_composition(formula_str):
    """Check if formula string can be parsed by pymatgen Composition"""
    try:
        c = Composition(formula_str)
        return len(c.elements) > 0
    except Exception:
        return False


if __name__ == '__main__':
    df_gap = load_bandgap_dataset()
    print("Band Gap Dataset shape:", df_gap.shape)
    print(df_gap.head())

    df_metal = load_metallicity_dataset()
    print("\nMetallicity Dataset shape:", df_metal.shape)
    print(df_metal.head())
