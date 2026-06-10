"""
01_data_and_baseline.py
Load Norman et al. 2019 via GEARS PertData and compute the additive baseline
on the same simulation split that GEARS will use.

Additive model: predicted combo(A,B) = ctrl + shift(A) + shift(B)
Evaluated by Pearson r on top 20 DE genes per perturbation.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GEARS'))

import numpy as np
import pandas as pd
import pickle
from scipy.stats import pearsonr
from gears import PertData

OUTPUT_DIR = "results"
DATA_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load data via GEARS ──────────────────────────────────────────────
print("Loading Norman dataset via GEARS PertData...")
print("  (First run downloads ~150 MB from Harvard Dataverse)")
pert_data = PertData(DATA_DIR)
pert_data.load(data_name='norman')
pert_data.prepare_split(split='simulation', seed=1)
pert_data.get_dataloader(batch_size=32, test_batch_size=128)

adata = pert_data.adata
subgroup = pert_data.subgroup
set2conditions = pert_data.set2conditions

print(f"Dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")
print(f"Split: train={len(set2conditions['train'])} | "
      f"val={len(set2conditions['val'])} | "
      f"test={len(set2conditions['test'])} conditions")

# ── 2. Compute mean expression per condition ─────────────────────────────
print("Computing per-condition mean expression...")
condition_means = {}
for cond in adata.obs['condition'].unique():
    mask = adata.obs['condition'] == cond
    condition_means[cond] = np.asarray(adata[mask].X.mean(axis=0)).flatten()

ctrl_mean = condition_means['ctrl']

# Mappings
pert2name = dict(adata.obs[['condition', 'condition_name']].drop_duplicates().values)
gene_id2idx = dict(zip(adata.var.index.values, range(len(adata.var))))
gene_names = list(adata.var['gene_name'].values)
de_genes_dict = adata.uns.get('rank_genes_groups_cov_all', {})

# ── 3. Identify test combos ─────────────────────────────────────────────
test_combos = [c for c in set2conditions['test']
               if c != 'ctrl' and '+' in c and 'ctrl' not in c]
print(f"Test combos: {len(test_combos)}")

# ── 4. Additive baseline ────────────────────────────────────────────────
print("Computing additive baseline predictions...")

def find_single(gene):
    """Find the single-gene perturbation condition string."""
    for fmt in [f"{gene}+ctrl", f"ctrl+{gene}"]:
        if fmt in condition_means:
            return fmt
    return None

results = []
for combo in test_combos:
    genes = [g for g in combo.split('+') if g != 'ctrl']
    if len(genes) != 2:
        continue

    ga, gb = genes
    sa, sb = find_single(ga), find_single(gb)

    if sa is None or sb is None or combo not in condition_means:
        print(f"  Skip {combo}: missing single-gene data")
        continue

    # Additive prediction: ctrl + shift_A + shift_B
    predicted = ctrl_mean + (condition_means[sa] - ctrl_mean) + \
                            (condition_means[sb] - ctrl_mean)
    actual = condition_means[combo]

    # DE gene indices (same genes GEARS uses internally)
    cname = pert2name.get(combo, combo)
    if cname in de_genes_dict:
        de_idx = [gene_id2idx[g] for g in de_genes_dict[cname][:20]
                  if g in gene_id2idx]
    else:
        de_idx = np.argsort(np.abs(actual - ctrl_mean))[-20:].tolist()

    if len(de_idx) < 2:
        continue

    r_de, p_de = pearsonr(predicted[de_idx], actual[de_idx])
    mse_de = float(np.mean((predicted[de_idx] - actual[de_idx]) ** 2))
    non_add = float(np.linalg.norm(actual - predicted) /
                    (np.linalg.norm(actual - ctrl_mean) + 1e-8))

    # Subgroup label from GEARS split
    sg = 'unknown'
    if subgroup and 'test_subgroup' in subgroup:
        for name, perts in subgroup['test_subgroup'].items():
            if combo in perts:
                sg = name
                break

    results.append({
        'condition': combo, 'gene_a': ga, 'gene_b': gb,
        'pearson_r_de20': r_de, 'mse_de20': mse_de,
        'non_additivity': non_add, 'subgroup': sg,
    })

results_df = pd.DataFrame(results)

# ── 5. Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Additive Baseline -- {len(results_df)} test combos")
print(f"{'='*50}")
print(f"  Mean Pearson r (top 20 DE): {results_df['pearson_r_de20'].mean():.4f}")
print(f"  Mean MSE      (top 20 DE): {results_df['mse_de20'].mean():.4f}")
for sg in sorted(results_df['subgroup'].unique()):
    sdf = results_df[results_df['subgroup'] == sg]
    print(f"  {sg}: r={sdf['pearson_r_de20'].mean():.4f}  (n={len(sdf)})")

# ── 6. Save ──────────────────────────────────────────────────────────────
results_df.to_csv(os.path.join(OUTPUT_DIR, 'additive_baseline_results.csv'),
                  index=False)

baseline_data = {
    'results_df': results_df,
    'condition_means': condition_means,
    'ctrl_mean': ctrl_mean,
    'subgroup': subgroup,
    'set2conditions': set2conditions,
    'pert2name': pert2name,
    'gene_id2idx': gene_id2idx,
    'gene_names': gene_names,
    'de_genes_dict': {k: list(v[:20]) for k, v in de_genes_dict.items()},
}
with open(os.path.join(OUTPUT_DIR, 'baseline_data.pkl'), 'wb') as f:
    pickle.dump(baseline_data, f)

print(f"\nSaved to {OUTPUT_DIR}/")
print("Done!")
