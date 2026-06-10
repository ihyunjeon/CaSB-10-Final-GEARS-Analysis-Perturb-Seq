"""
01_preprocessing.py
Preprocessing pipeline for Norman et al. 2019 CRISPRa dataset.
- Library-size normalization + log1p
- Top 5,000 HVG selection
- Per-perturbation mean expression shift relative to control
- Train/test split: hold out ~20 unseen two-gene combinations
"""

import pertpy as pp
import scanpy as sc
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

OUTPUT_DIR = "results"
DATA_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
adata = sc.read_h5ad("data/NormanWeissman2019_filtered.h5ad")
print(f"Loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")
print(f"Perturbation conditions: {adata.obs['perturbation'].nunique()}")

# ── 2. Identify perturbation types ──────────────────────────────────────────
# scPerturb uses 'nperts' column: 0=control, 1=single, 2=combo
# Combo perturbations use '_' as separator (e.g. 'CBL_CNN1')
perturbations = adata.obs['perturbation'].unique()
nperts_map = adata.obs.groupby('perturbation')['nperts'].first()

controls = [p for p in perturbations if nperts_map[p] == 0]
singles = [p for p in perturbations if nperts_map[p] == 1]
combos = [p for p in perturbations if nperts_map[p] == 2]

print(f"Controls: {len(controls)}, Singles: {len(singles)}, Combos: {len(combos)}")

# ── 3. Normalize ────────────────────────────────────────────────────────────
print("Normalizing...")
# Store raw counts
adata.layers['counts'] = adata.X.copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# ── 4. Select top 5,000 HVGs ────────────────────────────────────────────────
print("Selecting HVGs...")
sc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor='seurat_v3',
                            layer='counts')
adata_hvg = adata[:, adata.var['highly_variable']].copy()
print(f"After HVG selection: {adata_hvg.shape}")

# ── 5. Compute per-perturbation mean expression ────────────────────────────
print("Computing per-perturbation mean expression profiles...")

# Get control mean
control_mask = adata_hvg.obs['perturbation'].isin(controls)
control_mean = np.array(adata_hvg[control_mask].X.mean(axis=0)).flatten()

# Compute mean expression and shift for each perturbation
mean_expr = {}
mean_shift = {}

for pert in singles + combos:
    mask = adata_hvg.obs['perturbation'] == pert
    if mask.sum() == 0:
        continue
    expr = np.array(adata_hvg[mask].X.mean(axis=0)).flatten()
    mean_expr[pert] = expr
    mean_shift[pert] = expr - control_mean

print(f"Computed shifts for {len(mean_shift)} perturbations")

# ── 6. Train/test split ────────────────────────────────────────────────────
# Hold out ~20 unseen two-gene combinations
print("Creating train/test split...")

np.random.seed(42)
combo_list = [c for c in combos if c in mean_shift]
test_combos = list(np.random.choice(combo_list, size=min(20, len(combo_list)),
                                     replace=False))
train_combos = [c for c in combo_list if c not in test_combos]

print(f"Train combos: {len(train_combos)}, Test combos: {len(test_combos)}")
print(f"Test combos: {test_combos}")

# ── 7. Save processed data ─────────────────────────────────────────────────
print("Saving processed data...")

processed = {
    'control_mean': control_mean,
    'mean_expr': mean_expr,
    'mean_shift': mean_shift,
    'singles': singles,
    'combos': combo_list,
    'train_combos': train_combos,
    'test_combos': test_combos,
    'gene_names': adata_hvg.var_names.tolist(),
}

with open(os.path.join(DATA_DIR, 'processed_data.pkl'), 'wb') as f:
    pickle.dump(processed, f)

# Save the HVG-filtered adata for GEARS
adata_hvg.write(os.path.join(DATA_DIR, 'norman_hvg.h5ad'))

print("Done! Saved processed_data.pkl and norman_hvg.h5ad to data/")
