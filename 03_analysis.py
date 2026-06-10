"""
03_analysis.py
Load results from scripts 01 & 02, generate 6 presentation-ready figures:
  1. Training curves (loss + validation MSE per epoch)
  2. t-SNE of learned perturbation embeddings
  3. Subgroup bar chart (Additive vs GEARS by seen0/1/2)
  4. Per-perturbation scatter (additive r vs GEARS r)
  5. Example perturbation plot (box plot + model predictions)
  6. Non-additivity stratified analysis (GEARS advantage by interaction strength)
"""

import numpy as np
import pandas as pd
import pickle
import os
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "results"
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'font.family': 'sans-serif',
})

# ── Load saved data ─────────────────────────────────────────────────────
print("Loading results...")
baseline_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'additive_baseline_results.csv'))
history_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'training_history.csv'))

with open(os.path.join(OUTPUT_DIR, 'baseline_data.pkl'), 'rb') as f:
    bd = pickle.load(f)
with open(os.path.join(OUTPUT_DIR, 'gears_data.pkl'), 'rb') as f:
    gd = pickle.load(f)

condition_means = bd['condition_means']
ctrl_mean = bd['ctrl_mean']
pert2name = bd['pert2name']
gene_id2idx = bd['gene_id2idx']
gene_names = bd['gene_names']
de_genes_dict = bd['de_genes_dict']
subgroup = bd['subgroup']
set2conditions = bd['set2conditions']

test_pert_res = gd['test_pert_res']
gears_pred_means = gd['gears_pred_means']
test_res = gd['test_res']
pert_emb = gd['pert_emb_gnn']
pert_names = gd['pert_names']

# ── Build comparison dataframe ───────────────────────────────────────────
comparison = []
for _, row in baseline_df.iterrows():
    c = row['condition']
    if c in test_pert_res and 'pearson_de' in test_pert_res[c]:
        comparison.append({
            'condition': c,
            'subgroup': row['subgroup'],
            'additive_r': row['pearson_r_de20'],
            'gears_r': test_pert_res[c]['pearson_de'],
            'additive_mse': row['mse_de20'],
            'gears_mse': test_pert_res[c].get('mse_de', np.nan),
            'non_additivity': row['non_additivity'],
        })
comp_df = pd.DataFrame(comparison)
comp_df['improvement'] = comp_df['gears_r'] - comp_df['additive_r']
comp_df['mse_improvement'] = comp_df['additive_mse'] - comp_df['gears_mse']  # positive = GEARS better
print(f"Matched {len(comp_df)} test combos for comparison")
print(f"  Pearson r:  Additive={comp_df['additive_r'].mean():.4f}, "
      f"GEARS={comp_df['gears_r'].mean():.4f}")
print(f"  MSE (DE20): Additive={comp_df['additive_mse'].mean():.4f}, "
      f"GEARS={comp_df['gears_mse'].mean():.4f}\n")


# ═════════════════════════════════════════════════════════════════════════
# Figure 1: Training Curves
# ═════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(history_df['epoch'], history_df['train_loss'], 'o-',
         color='#1565C0', markersize=4, linewidth=1.5, label='Training loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history_df['epoch'], history_df['val_mse_de'], 's-',
         color='#C62828', markersize=4, linewidth=1.5, label='Top 20 DE genes')
ax2.plot(history_df['epoch'], history_df['val_mse'], '^-',
         color='#2E7D32', markersize=4, linewidth=1.5, label='All genes')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE')
ax2.set_title('Validation MSE')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_training_curves.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig1_training_curves.png")


# ═════════════════════════════════════════════════════════════════════════
# Figure 2: t-SNE of Perturbation Embeddings
# ═════════════════════════════════════════════════════════════════════════

# Identify genes actually perturbed in the experiment
all_conds = []
for split in ['train', 'val', 'test']:
    all_conds.extend(set2conditions.get(split, []))

perturbed_genes = set()
test_combo_genes = set()
train_combo_genes = set()

for c in all_conds:
    for g in c.split('+'):
        if g != 'ctrl':
            perturbed_genes.add(g)

for c in set2conditions.get('test', []):
    if 'ctrl' not in c and '+' in c:
        for g in c.split('+'):
            test_combo_genes.add(g)

for c in set2conditions.get('train', []):
    if 'ctrl' not in c and '+' in c:
        for g in c.split('+'):
            train_combo_genes.add(g)

# Map to embedding indices
pert_name_list = list(pert_names) if not isinstance(pert_names, list) else pert_names
pert_idx_map = {name: i for i, name in enumerate(pert_name_list)}
active_genes = [g for g in perturbed_genes if g in pert_idx_map]
active_idx = [pert_idx_map[g] for g in active_genes]
active_emb = pert_emb[active_idx]

# Categorize each gene
categories = []
for g in active_genes:
    if g in test_combo_genes:
        categories.append('Test combo gene')
    elif g in train_combo_genes:
        categories.append('Train combo gene')
    else:
        categories.append('Single pert only')

# Run t-SNE
perp = max(5, min(30, len(active_genes) // 3))
tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
emb_2d = tsne.fit_transform(active_emb)

fig, ax = plt.subplots(figsize=(9, 7))
colors_map = {
    'Test combo gene': '#C62828',
    'Train combo gene': '#1565C0',
    'Single pert only': '#BDBDBD',
}
# Draw order: gray first, then blue, then red on top
for cat in ['Single pert only', 'Train combo gene', 'Test combo gene']:
    mask = np.array([c == cat for c in categories])
    if mask.sum() == 0:
        continue
    sz = 20 if cat == 'Single pert only' else 45
    alpha = 0.4 if cat == 'Single pert only' else 0.85
    ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
               c=colors_map[cat], label=cat, s=sz, alpha=alpha,
               edgecolors='black', linewidth=0.3)

# Annotate test combo genes
for i, g in enumerate(active_genes):
    if g in test_combo_genes:
        ax.annotate(g, (emb_2d[i, 0], emb_2d[i, 1]),
                    fontsize=5.5, alpha=0.75,
                    xytext=(3, 3), textcoords='offset points')

ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('GEARS Perturbation Embeddings (post-GNN)')
ax.legend(loc='best', framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_tsne_embeddings.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig2_tsne_embeddings.png")


# ═════════════════════════════════════════════════════════════════════════
# Figure 3: Subgroup Comparison Bar Chart (Pearson r + MSE)
# ═════════════════════════════════════════════════════════════════════════
subgroups = sorted([s for s in comp_df['subgroup'].unique() if s != 'unknown'])
if 'unknown' in comp_df['subgroup'].values:
    subgroups.append('unknown')

means_add_r, means_gears_r = [], []
sems_add_r, sems_gears_r = [], []
means_add_mse, means_gears_mse = [], []
sems_add_mse, sems_gears_mse = [], []
counts = []

for sg in subgroups:
    sdf = comp_df[comp_df['subgroup'] == sg]
    means_add_r.append(sdf['additive_r'].mean())
    means_gears_r.append(sdf['gears_r'].mean())
    sems_add_r.append(sdf['additive_r'].sem())
    sems_gears_r.append(sdf['gears_r'].sem())
    means_add_mse.append(sdf['additive_mse'].mean())
    means_gears_mse.append(sdf['gears_mse'].mean())
    sems_add_mse.append(sdf['additive_mse'].sem())
    sems_gears_mse.append(sdf['gears_mse'].sem())
    counts.append(len(sdf))

# Append overall
subgroups.append('Overall')
means_add_r.append(comp_df['additive_r'].mean())
means_gears_r.append(comp_df['gears_r'].mean())
sems_add_r.append(comp_df['additive_r'].sem())
sems_gears_r.append(comp_df['gears_r'].sem())
means_add_mse.append(comp_df['additive_mse'].mean())
means_gears_mse.append(comp_df['gears_mse'].mean())
sems_add_mse.append(comp_df['additive_mse'].sem())
sems_gears_mse.append(comp_df['gears_mse'].sem())
counts.append(len(comp_df))

x = np.arange(len(subgroups))
w = 0.32

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Panel A: Pearson r (higher is better)
ax1.bar(x - w/2, means_add_r, w, yerr=sems_add_r, capsize=4,
        label='Additive Baseline', color='#42A5F5', edgecolor='black', linewidth=0.5)
ax1.bar(x + w/2, means_gears_r, w, yerr=sems_gears_r, capsize=4,
        label='GEARS', color='#EF5350', edgecolor='black', linewidth=0.5)
ax1.set_ylabel('Mean Pearson r (top 20 DE genes)')
ax1.set_title('Pearson Correlation by Subgroup')
ax1.set_xticks(x)
ax1.set_xticklabels([f"{s}\n(n={c})" for s, c in zip(subgroups, counts)])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(bottom=0)

# Panel B: MSE (lower is better)
ax2.bar(x - w/2, means_add_mse, w, yerr=sems_add_mse, capsize=4,
        label='Additive Baseline', color='#42A5F5', edgecolor='black', linewidth=0.5)
ax2.bar(x + w/2, means_gears_mse, w, yerr=sems_gears_mse, capsize=4,
        label='GEARS', color='#EF5350', edgecolor='black', linewidth=0.5)
ax2.set_ylabel('Mean MSE (top 20 DE genes)')
ax2.set_title('MSE by Subgroup (lower is better)')
ax2.set_xticks(x)
ax2.set_xticklabels([f"{s}\n(n={c})" for s, c in zip(subgroups, counts)])
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_subgroup_comparison.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig3_subgroup_comparison.png")


# ═════════════════════════════════════════════════════════════════════════
# Figure 4: Per-Perturbation Scatter
# ═════════════════════════════════════════════════════════════════════════
sg_colors = {
    'combo_seen0': '#C62828', 'combo_seen1': '#EF6C00',
    'combo_seen2': '#2E7D32', 'unknown': '#757575',
}

fig, ax = plt.subplots(figsize=(7, 7))
for sg in comp_df['subgroup'].unique():
    mask = comp_df['subgroup'] == sg
    ax.scatter(comp_df.loc[mask, 'additive_r'],
               comp_df.loc[mask, 'gears_r'],
               c=sg_colors.get(sg, '#757575'), label=sg,
               s=65, edgecolors='black', linewidth=0.5, alpha=0.85)

# Diagonal parity line
lo = min(comp_df['additive_r'].min(), comp_df['gears_r'].min()) - 0.05
hi = max(comp_df['additive_r'].max(), comp_df['gears_r'].max()) + 0.05
ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='y = x')
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)

ax.set_xlabel('Additive Baseline  Pearson r')
ax.set_ylabel('GEARS  Pearson r')
ax.set_title('Per-Perturbation Comparison (top 20 DE)')
ax.legend(loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_per_pert_scatter.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig4_per_pert_scatter.png")


# ═════════════════════════════════════════════════════════════════════════
# Figure 5: Example Perturbation Plot
# ═════════════════════════════════════════════════════════════════════════

# Pick the combo with the largest GEARS improvement
best_idx = comp_df['improvement'].idxmax()
best_combo = comp_df.loc[best_idx, 'condition']
print(f"\nExample perturbation: {best_combo}")
print(f"  Additive r = {comp_df.loc[best_idx, 'additive_r']:.3f}")
print(f"  GEARS    r = {comp_df.loc[best_idx, 'gears_r']:.3f}")

# DE gene indices & names
combo_name = pert2name.get(best_combo, best_combo)
if combo_name in de_genes_dict:
    de_gene_ids = de_genes_dict[combo_name][:20]
    de_idx = [gene_id2idx[g] for g in de_gene_ids if g in gene_id2idx]
else:
    de_idx = np.argsort(np.abs(condition_means[best_combo] - ctrl_mean))[-20:].tolist()
de_gene_labels = [gene_names[i] for i in de_idx]

# Cell-level truth distribution (change from control)
p_mask = test_res['pert_cat'] == best_combo
if p_mask.sum() == 0:
    print(f"  WARNING: {best_combo} not found in test_res, trying another...")
    # Fall back to second-best
    best_idx = comp_df.nlargest(2, 'improvement').index[1]
    best_combo = comp_df.loc[best_idx, 'condition']
    p_mask = test_res['pert_cat'] == best_combo
    combo_name = pert2name.get(best_combo, best_combo)
    if combo_name in de_genes_dict:
        de_gene_ids = de_genes_dict[combo_name][:20]
        de_idx = [gene_id2idx[g] for g in de_gene_ids if g in gene_id2idx]
    else:
        de_idx = np.argsort(np.abs(condition_means[best_combo] - ctrl_mean))[-20:].tolist()
    de_gene_labels = [gene_names[i] for i in de_idx]

truth_cells = test_res['truth'][p_mask][:, de_idx]
truth_delta = truth_cells - ctrl_mean[de_idx]

# GEARS predicted change
gears_delta = gears_pred_means[best_combo][de_idx] - ctrl_mean[de_idx]

# Additive predicted change
genes = [g for g in best_combo.split('+') if g != 'ctrl']
ga, gb = genes[0], genes[1]
sa = next((f for f in [f"{ga}+ctrl", f"ctrl+{ga}"] if f in condition_means), None)
sb = next((f for f in [f"{gb}+ctrl", f"ctrl+{gb}"] if f in condition_means), None)
if sa and sb:
    additive_delta = ((condition_means[sa] - ctrl_mean) +
                      (condition_means[sb] - ctrl_mean))[de_idx]
else:
    additive_delta = np.zeros(len(de_idx))

# Plot
fig, ax = plt.subplots(figsize=(14, 5))
bp = ax.boxplot(truth_delta, showfliers=False,
                medianprops=dict(color='black', linewidth=1.5),
                boxprops=dict(facecolor='#E0E0E0'), patch_artist=True)

positions = list(range(1, len(de_idx) + 1))
ax.scatter(positions, gears_delta, color='#C62828', s=70, zorder=5,
           label=f'GEARS (r={comp_df.loc[best_idx, "gears_r"]:.3f})',
           edgecolors='black', linewidth=0.5)
ax.scatter(positions, additive_delta, color='#1565C0', s=70, zorder=5,
           marker='^',
           label=f'Additive (r={comp_df.loc[best_idx, "additive_r"]:.3f})',
           edgecolors='black', linewidth=0.5)

ax.axhline(0, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xticks(positions)
ax.set_xticklabels(de_gene_labels, rotation=90, fontsize=8)
ax.set_ylabel('Expression Change from Control')
ax.set_title(f'Perturbation: {best_combo}  (top 20 DE genes)')
ax.legend(loc='best', framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_example_perturbation.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig5_example_perturbation.png")


# ═════════════════════════════════════════════════════════════════════════
# Figure 6: Non-Additivity Stratified Analysis
# ═════════════════════════════════════════════════════════════════════════
# Split combos into terciles by non-additivity score.
# High non-additivity = strong synergistic/antagonistic interaction
# = where additive model SHOULD fail and GEARS should shine.

na_valid = comp_df.dropna(subset=['non_additivity', 'gears_mse', 'additive_mse']).copy()
if len(na_valid) >= 9:
    na_valid['na_rank'] = pd.qcut(na_valid['non_additivity'], q=3,
                                   labels=['Low\n(additive)', 'Medium', 'High\n(synergistic/\nantagonistic)'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Bar chart — MSE by non-additivity tercile (paper's primary metric)
    ax = axes[0]
    terciles = ['Low\n(additive)', 'Medium', 'High\n(synergistic/\nantagonistic)']
    t_means_add, t_means_gears = [], []
    t_sems_add, t_sems_gears = [], []
    t_counts = []
    for t in terciles:
        tdf = na_valid[na_valid['na_rank'] == t]
        t_means_add.append(tdf['additive_mse'].mean())
        t_means_gears.append(tdf['gears_mse'].mean())
        t_sems_add.append(tdf['additive_mse'].sem())
        t_sems_gears.append(tdf['gears_mse'].sem())
        t_counts.append(len(tdf))

    x = np.arange(len(terciles))
    w = 0.32
    ax.bar(x - w/2, t_means_add, w, yerr=t_sems_add, capsize=4,
           label='Additive', color='#42A5F5', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, t_means_gears, w, yerr=t_sems_gears, capsize=4,
           label='GEARS', color='#EF5350', edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Mean MSE (top 20 DE genes)')
    ax.set_title('MSE by Interaction Strength (lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={c})" for t, c in zip(terciles, t_counts)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel B: Scatter — non-additivity vs GEARS MSE advantage
    # Positive = GEARS has lower MSE (better)
    ax = axes[1]
    mse_adv = na_valid['additive_mse'] - na_valid['gears_mse']
    scatter_valid = na_valid['non_additivity'].notna() & mse_adv.notna()
    na_x = na_valid.loc[scatter_valid, 'non_additivity']
    adv_y = mse_adv[scatter_valid]

    ax.scatter(na_x, adv_y, c='#5E35B1', s=50, alpha=0.7,
               edgecolors='black', linewidth=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Trend line
    if len(na_x) > 2:
        z = np.polyfit(na_x.values, adv_y.values, 1)
        x_line = np.linspace(na_x.min(), na_x.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), 'r-', linewidth=2, alpha=0.7,
                label=f'Trend (slope={z[0]:.3f})')

        r_na, p_na = pearsonr(na_x.values, adv_y.values)
        ax.set_title(f'GEARS MSE Advantage vs Interaction Strength\n(r={r_na:.3f}, p={p_na:.4f})')
    else:
        ax.set_title('GEARS MSE Advantage vs Interaction Strength')

    ax.set_xlabel('Non-Additivity Score')
    ax.set_ylabel('GEARS MSE Advantage\n(Additive MSE - GEARS MSE, positive = GEARS better)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_non_additivity_analysis.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig6_non_additivity_analysis.png")
else:
    print(f"WARN: Only {len(na_valid)} combos with non-additivity scores, skipping fig6")


# ═════════════════════════════════════════════════════════════════════════
# Figure 7: Unseen Gene Evaluation — GEARS's unique advantage
# ═════════════════════════════════════════════════════════════════════════
# For genes never individually perturbed during training, the additive
# baseline CANNOT make predictions (no measured single-gene effect).
# GEARS can still predict via the GO knowledge graph.
# Baseline here: "no perturbation" (predict ctrl expression).

unseen_singles = []
if subgroup and 'test_subgroup' in subgroup:
    unseen_singles = subgroup['test_subgroup'].get('unseen_single', [])

if len(unseen_singles) > 0:
    unseen_results = []
    for pert in unseen_singles:
        if pert not in test_pert_res:
            continue

        # Get truth and GEARS prediction from cell-level test data
        p_mask = test_res['pert_cat'] == pert
        if p_mask.sum() == 0:
            continue

        truth_mean = test_res['truth'][p_mask].mean(axis=0)
        gears_mean = test_res['pred'][p_mask].mean(axis=0)

        # DE gene indices for this perturbation
        pname = pert2name.get(pert, pert)
        if pname in de_genes_dict:
            de_idx = [gene_id2idx[g] for g in de_genes_dict[pname][:20]
                      if g in gene_id2idx]
        else:
            de_idx = np.argsort(np.abs(truth_mean - ctrl_mean))[-20:].tolist()

        if len(de_idx) < 2:
            continue

        # GEARS metrics on DE genes
        gears_mse_de = float(np.mean((gears_mean[de_idx] - truth_mean[de_idx]) ** 2))
        try:
            gears_r_de = float(pearsonr(gears_mean[de_idx], truth_mean[de_idx])[0])
        except:
            gears_r_de = np.nan

        # No-perturbation baseline: predict ctrl expression (best guess without data)
        nop_mse_de = float(np.mean((ctrl_mean[de_idx] - truth_mean[de_idx]) ** 2))
        try:
            nop_r_de = float(pearsonr(ctrl_mean[de_idx], truth_mean[de_idx])[0])
        except:
            nop_r_de = np.nan

        gene_name = pert.replace('+ctrl', '').replace('ctrl+', '')
        unseen_results.append({
            'perturbation': pert,
            'gene': gene_name,
            'gears_r': gears_r_de,
            'gears_mse': gears_mse_de,
            'nop_r': nop_r_de,
            'nop_mse': nop_mse_de,
        })

    unseen_df = pd.DataFrame(unseen_results)

    if len(unseen_df) > 0:
        # Count how often GEARS beats no-perturbation baseline
        gears_wins_mse = (unseen_df['gears_mse'] < unseen_df['nop_mse']).sum()

        print(f"\nUnseen single-gene perturbations: {len(unseen_df)}")
        print(f"  GEARS:            MSE={unseen_df['gears_mse'].mean():.4f}, "
              f"r={unseen_df['gears_r'].mean():.4f}")
        print(f"  No-perturbation:  MSE={unseen_df['nop_mse'].mean():.4f}, "
              f"r={unseen_df['nop_r'].mean():.4f}")
        print(f"  GEARS wins (lower MSE): {gears_wins_mse}/{len(unseen_df)} "
              f"({100*gears_wins_mse/len(unseen_df):.0f}%)")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

        # Panel A: Paired comparison — MSE for each unseen gene
        sorted_df = unseen_df.sort_values('nop_mse', ascending=False).reset_index(drop=True)
        y_pos = np.arange(len(sorted_df))

        ax1.barh(y_pos, sorted_df['nop_mse'], height=0.4, left=0,
                 color='#BDBDBD', edgecolor='black', linewidth=0.3,
                 label='No Perturbation')
        ax1.barh(y_pos - 0.4, sorted_df['gears_mse'], height=0.4, left=0,
                 color='#EF5350', edgecolor='black', linewidth=0.3,
                 label='GEARS')
        ax1.set_yticks(y_pos - 0.2)
        ax1.set_yticklabels(sorted_df['gene'], fontsize=6)
        ax1.set_xlabel('MSE (top 20 DE genes)')
        ax1.set_title(f'Unseen Gene Predictions (n={len(unseen_df)})\n'
                      f'Additive baseline cannot predict these')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()

        # Panel B: MSE reduction scatter
        ax2.scatter(unseen_df['nop_mse'], unseen_df['gears_mse'],
                    c='#EF5350', s=55, edgecolors='black', linewidth=0.5, alpha=0.8)
        lo = 0
        hi = max(unseen_df['nop_mse'].max(), unseen_df['gears_mse'].max()) * 1.1
        ax2.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='y = x (equal)')
        ax2.set_xlabel('No-Perturbation MSE')
        ax2.set_ylabel('GEARS MSE')
        ax2.set_title(f'GEARS vs No-Perturbation Baseline\n'
                      f'(below diagonal = GEARS better)\n'
                      f'GEARS wins {gears_wins_mse}/{len(unseen_df)} genes')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(lo, hi)

        # Annotate top GEARS wins
        best_wins = unseen_df.copy()
        best_wins['mse_reduction'] = best_wins['nop_mse'] - best_wins['gears_mse']
        for _, row in best_wins.nlargest(3, 'mse_reduction').iterrows():
            ax2.annotate(row['gene'], (row['nop_mse'], row['gears_mse']),
                         fontsize=7, xytext=(5, 5), textcoords='offset points')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_unseen_gene_evaluation.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()
        print("Saved fig7_unseen_gene_evaluation.png")

        unseen_df.to_csv(os.path.join(OUTPUT_DIR, 'unseen_gene_results.csv'), index=False)
    else:
        print("No unseen_single perturbations matched in test results")
else:
    print("No unseen_single perturbations found in test split")


# ═════════════════════════════════════════════════════════════════════════
# Save comparison table & print summary
# ═════════════════════════════════════════════════════════════════════════
comp_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)

print(f"\n{'='*50}")
print("All figures saved to results/:")
for i in range(1, 8):
    name = ['training_curves', 'tsne_embeddings', 'subgroup_comparison',
            'per_pert_scatter', 'example_perturbation', 'non_additivity_analysis',
            'unseen_gene_evaluation'][i-1]
    print(f"  fig{i}_{name}.png")
print(f"\nComparison table: results/model_comparison.csv")
print(f"\nOverall Pearson r: Additive={comp_df['additive_r'].mean():.4f}  |  "
      f"GEARS={comp_df['gears_r'].mean():.4f}")
print(f"Overall MSE (DE20): Additive={comp_df['additive_mse'].mean():.4f}  |  "
      f"GEARS={comp_df['gears_mse'].mean():.4f}")
print("Done!")
