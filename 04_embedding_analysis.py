"""
04_embedding_analysis.py
Analyze GEARS perturbation embeddings:
- t-SNE visualization colored by perturbation type / pathway
- Relative Neighborhood Score (RNS) metric
"""

import torch
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from gears import PertData, GEARS

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load trained GEARS model ───────────────────────────────────────────
print("Loading trained GEARS model...")
if torch.cuda.is_available():
    device = 'cuda:0'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

pert_data = PertData('./data')
pert_data.load(data_name='norman')
pert_data.prepare_split(split='simulation', seed=1)
pert_data.get_dataloader(batch_size=32, test_batch_size=128)

gears_model = GEARS(pert_data, device=device)
gears_model.load_pretrained(os.path.join(OUTPUT_DIR, 'gears_model'))

# ── 2. Extract perturbation embeddings ─────────────────────────────────────
print("Extracting perturbation embeddings...")

# Access the learned gene embeddings from the model
# The exact attribute depends on GEARS version; adapt as needed
model = gears_model.model
model.eval()

# Get gene-level perturbation embeddings
gene_names = pert_data.gene_names  # or pert_data.pert_names
emb_matrix = model.gene_emb.weight.detach().cpu().numpy()

print(f"Embedding matrix shape: {emb_matrix.shape}")

# ── 3. t-SNE visualization ────────────────────────────────────────────────
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
embeddings_2d = tsne.fit_transform(emb_matrix)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                     alpha=0.6, s=30, edgecolors='black', linewidth=0.3)

# Label a few notable genes
# Adjust these based on what's actually in the dataset
notable_genes = ['CEBPA', 'TP73', 'FOXA3', 'SPI1', 'UBASH3B']
for gene in notable_genes:
    if gene in gene_names:
        idx = list(gene_names).index(gene)
        ax.annotate(gene, (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                    fontsize=8, fontweight='bold')

ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE of GEARS Perturbation Embeddings')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_embeddings.png'), dpi=150)
print(f"t-SNE plot saved to {OUTPUT_DIR}/tsne_embeddings.png")

# ── 4. Relative Neighborhood Score (RNS) ──────────────────────────────────
print("\nComputing Relative Neighborhood Score (RNS)...")

def compute_rns(embeddings, labels, n_neighbors=10):
    """
    Compute Relative Neighborhood Score.

    For each point, check what fraction of its k-nearest neighbors
    share the same label. Compare to random baseline.

    RNS = (observed_fraction - expected_fraction) / (1 - expected_fraction)

    RNS=1 means perfect clustering by label, RNS=0 means random.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 for self
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    # Remove self-neighbor
    indices = indices[:, 1:]

    unique_labels = np.unique(labels)
    label_counts = pd.Series(labels).value_counts()

    observed_fracs = []
    for i in range(len(embeddings)):
        neighbor_labels = labels[indices[i]]
        same_label = np.sum(neighbor_labels == labels[i])
        observed_fracs.append(same_label / n_neighbors)

    observed = np.mean(observed_fracs)

    # Expected fraction under random assignment
    expected = np.sum((label_counts / len(labels)) ** 2)

    rns = (observed - expected) / (1 - expected) if expected < 1 else 0
    return rns, observed, expected


# Create labels based on gene function categories (simplified)
# In practice, you'd use GO annotations or pathway membership
# Here we use a placeholder based on the perturbation data
print("(Using perturbation type as label for RNS)")

# For a meaningful RNS, we need category labels for each gene
# This is a simplified version - enhance with GO annotations if available
labels = np.array(['gene'] * len(gene_names))  # placeholder

# If we have combo genes, label them differently
combo_genes = set()
for combo in pert_data.pert_names:
    if '_' in combo and combo.count('_') == 1:
        for g in combo.split('_'):
            combo_genes.add(g.strip())

for i, gene in enumerate(gene_names):
    if gene in combo_genes:
        labels[i] = 'combo_participant'

rns, observed, expected = compute_rns(emb_matrix, labels, n_neighbors=10)
print(f"RNS: {rns:.4f} (observed: {observed:.4f}, expected: {expected:.4f})")

# ── 5. Save embedding data ────────────────────────────────────────────────
embedding_data = {
    'gene_names': list(gene_names),
    'embeddings': emb_matrix,
    'tsne_2d': embeddings_2d,
    'rns': rns,
}
with open(os.path.join(OUTPUT_DIR, 'embedding_analysis.pkl'), 'wb') as f:
    pickle.dump(embedding_data, f)

print(f"\nEmbedding data saved to {OUTPUT_DIR}/embedding_analysis.pkl")
print("Done!")
