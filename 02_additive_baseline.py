"""
02_additive_baseline.py
Additive baseline: predicted combo A+B effect = shift(A) + shift(B).
Evaluated by Pearson correlation on top 20 DE genes per perturbation.
"""

import numpy as np
import pandas as pd
import pickle
import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load processed data ──────────────────────────────────────────────────
print("Loading processed data...")
with open('data/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

mean_shift = data['mean_shift']
test_combos = data['test_combos']
gene_names = data['gene_names']

# ── 2. Identify top 20 DE genes per perturbation ───────────────────────────
def get_top_de_genes(shift, n=20):
    """Return indices of top n DE genes by absolute shift magnitude."""
    return np.argsort(np.abs(shift))[-n:]

# ── 3. Additive predictions ────────────────────────────────────────────────
print("Running additive baseline on test combos...")
results = []

for combo in test_combos:
    genes = combo.split('_')
    if len(genes) != 2:
        print(f"  Skipping {combo} (not a 2-gene combo)")
        continue

    gene_a, gene_b = genes[0], genes[1]

    # Check both genes have single-gene data
    if gene_a not in mean_shift or gene_b not in mean_shift:
        print(f"  Skipping {combo}: missing single-gene data "
              f"(A={gene_a in mean_shift}, B={gene_b in mean_shift})")
        continue

    # Additive prediction
    predicted = mean_shift[gene_a] + mean_shift[gene_b]
    actual = mean_shift[combo]

    # Get top 20 DE genes from actual combo
    top_de_idx = get_top_de_genes(actual, n=20)

    # Pearson r on top 20 DE genes
    r, pval = pearsonr(predicted[top_de_idx], actual[top_de_idx])

    # Also compute full-gene Pearson r
    r_full, _ = pearsonr(predicted, actual)

    # Compute non-additivity score (how much combo deviates from sum)
    non_additivity = np.linalg.norm(actual - predicted) / np.linalg.norm(actual)

    results.append({
        'combo': combo,
        'gene_a': gene_a,
        'gene_b': gene_b,
        'pearson_r_top20': r,
        'pval_top20': pval,
        'pearson_r_full': r_full,
        'non_additivity': non_additivity,
    })

results_df = pd.DataFrame(results)
print(f"\nResults for {len(results_df)} test combos:")
print(results_df[['combo', 'pearson_r_top20', 'pearson_r_full', 'non_additivity']]
      .to_string(index=False))

# ── 4. Summary statistics ──────────────────────────────────────────────────
print(f"\n--- Additive Baseline Summary ---")
print(f"Mean Pearson r (top 20 DE): {results_df['pearson_r_top20'].mean():.4f}")
print(f"Median Pearson r (top 20 DE): {results_df['pearson_r_top20'].median():.4f}")
print(f"Mean Pearson r (all genes): {results_df['pearson_r_full'].mean():.4f}")
print(f"Mean non-additivity score: {results_df['non_additivity'].mean():.4f}")

# ── 5. Visualize ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Distribution of Pearson r
axes[0].hist(results_df['pearson_r_top20'], bins=15, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Pearson r (top 20 DE genes)')
axes[0].set_ylabel('Count')
axes[0].set_title('Additive Baseline: Pearson r Distribution')
axes[0].axvline(results_df['pearson_r_top20'].mean(), color='red',
                linestyle='--', label=f"Mean={results_df['pearson_r_top20'].mean():.3f}")
axes[0].legend()

# Pearson r vs non-additivity
axes[1].scatter(results_df['non_additivity'], results_df['pearson_r_top20'],
                alpha=0.7, edgecolors='black')
axes[1].set_xlabel('Non-additivity score')
axes[1].set_ylabel('Pearson r (top 20 DE genes)')
axes[1].set_title('Prediction Quality vs Non-additivity')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'additive_baseline_results.png'), dpi=150)
print(f"\nFigure saved to {OUTPUT_DIR}/additive_baseline_results.png")

# Save results
results_df.to_csv(os.path.join(OUTPUT_DIR, 'additive_baseline_results.csv'), index=False)
print(f"Results saved to {OUTPUT_DIR}/additive_baseline_results.csv")
