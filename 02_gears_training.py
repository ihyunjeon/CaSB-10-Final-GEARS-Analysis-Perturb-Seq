"""
02_gears_training.py
Train GEARS on Norman et al. 2019 (simulation split).
Custom training loop tracks per-epoch metrics for training curve plots.
Evaluates on held-out test combos and extracts perturbation embeddings.

Designed for Hoffman2 GPU (CUDA) but falls back to MPS/CPU.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GEARS'))

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.optim.lr_scheduler import StepLR

from gears import PertData, GEARS
from gears.inference import evaluate, compute_metrics
from gears.utils import loss_fct

OUTPUT_DIR = "results"
DATA_DIR = "data"
EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 1e-5
SCHEDULER_GAMMA = 0.95
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Device ────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = 'cuda:0'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"Device: {device}")
if device.startswith('cuda'):
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── 2. Load data (uses cached splits from script 01) ────────────────────
print("Loading data via GEARS PertData...")
pert_data = PertData(DATA_DIR)
pert_data.load(data_name='norman')
pert_data.prepare_split(split='simulation', seed=1)
pert_data.get_dataloader(batch_size=32, test_batch_size=128)

# ── 3. Initialize GEARS model ───────────────────────────────────────────
print("Initializing GEARS model...")
gears_model = GEARS(pert_data, device=device)
gears_model.model_initialize(hidden_size=64)
print(f"  Genes: {gears_model.num_genes}  Perturbations: {gears_model.num_perts}")

# ── 4. Training loop with history ───────────────────────────────────────
print(f"\nTraining GEARS ({EPOCHS} epochs, gamma={SCHEDULER_GAMMA})...")
train_loader = gears_model.dataloader['train_loader']
val_loader = gears_model.dataloader['val_loader']

model = gears_model.model
best_model = deepcopy(model)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = StepLR(optimizer, step_size=1, gamma=SCHEDULER_GAMMA)

history = {
    'epoch': [], 'train_loss': [],
    'val_mse': [], 'val_mse_de': [],
    'val_pearson': [], 'val_pearson_de': [],
}
min_val = np.inf

for epoch in range(EPOCHS):
    model.train()
    epoch_losses = []

    for batch in train_loader:
        batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fct(pred, batch.y, batch.pert,
                        ctrl=gears_model.ctrl_expression,
                        dict_filter=gears_model.dict_filter,
                        direction_lambda=gears_model.config['direction_lambda'])
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()
        epoch_losses.append(loss.item())

    scheduler.step()

    val_res = evaluate(val_loader, model, False, device)
    val_m, _ = compute_metrics(val_res)

    history['epoch'].append(epoch + 1)
    history['train_loss'].append(float(np.mean(epoch_losses)))
    history['val_mse'].append(float(val_m['mse']))
    history['val_mse_de'].append(float(val_m['mse_de']))
    history['val_pearson'].append(float(val_m['pearson']))
    history['val_pearson_de'].append(float(val_m['pearson_de']))

    lr_now = optimizer.param_groups[0]['lr']
    print(f"  Epoch {epoch+1:2d}/{EPOCHS}:  loss={np.mean(epoch_losses):.4f}  "
          f"val_MSE_DE={val_m['mse_de']:.4f}  val_r_DE={val_m['pearson_de']:.4f}  "
          f"lr={lr_now:.6f}")

    if val_m['mse_de'] < min_val:
        min_val = val_m['mse_de']
        best_model = deepcopy(model)
        print(f"    ↑ New best (val_MSE_DE={min_val:.4f})")

gears_model.best_model = best_model
gears_model.model = model

# Save model checkpoint
model_path = os.path.join(OUTPUT_DIR, 'gears_model')
gears_model.save_model(model_path)
print(f"Model saved to {model_path}/")

# ── 5. Test evaluation ──────────────────────────────────────────────────
print("\nEvaluating on test set...")
test_loader = gears_model.dataloader['test_loader']
test_res = evaluate(test_loader, best_model, False, device)
test_metrics, test_pert_res = compute_metrics(test_res)

print(f"  Overall MSE:       {test_metrics['mse']:.4f}")
print(f"  Top 20 DE MSE:     {test_metrics['mse_de']:.4f}")
print(f"  Overall Pearson:   {test_metrics['pearson']:.4f}")
print(f"  Top 20 DE Pearson: {test_metrics['pearson_de']:.4f}")

# Per-perturbation mean predictions
gears_pred_means = {}
for pert in np.unique(test_res['pert_cat']):
    idx = np.where(test_res['pert_cat'] == pert)[0]
    gears_pred_means[pert] = test_res['pred'][idx].mean(axis=0)

# ── 6. Extract embeddings ───────────────────────────────────────────────
print("Extracting perturbation embeddings...")
best_model.eval()

# Raw perturbation embeddings
pert_emb_raw = best_model.pert_emb.weight.detach().cpu().numpy()

# Post-GNN embeddings (augmented by gene ontology graph)
with torch.no_grad():
    pe = best_model.pert_emb(
        torch.arange(gears_model.num_perts).to(device))
    for i, layer in enumerate(best_model.sim_layers):
        pe = layer(pe, best_model.G_sim, best_model.G_sim_weight)
        if i < len(best_model.sim_layers) - 1:
            pe = pe.relu()
    pert_emb_gnn = pe.cpu().numpy()

print(f"  Perturbation embeddings: {pert_emb_gnn.shape}")

# ── 7. Save everything ──────────────────────────────────────────────────
history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(OUTPUT_DIR, 'training_history.csv'), index=False)

# Filter out ctrl cells from test_res to keep file size manageable
non_ctrl = test_res['pert_cat'] != 'ctrl'
test_res_save = {k: v[non_ctrl] for k, v in test_res.items()}

with open(os.path.join(OUTPUT_DIR, 'gears_data.pkl'), 'wb') as f:
    pickle.dump({
        'test_metrics': test_metrics,
        'test_pert_res': test_pert_res,
        'gears_pred_means': gears_pred_means,
        'test_res': test_res_save,
        'pert_emb_raw': pert_emb_raw,
        'pert_emb_gnn': pert_emb_gnn,
        'pert_names': gears_model.pert_list,
        'gene_list': gears_model.gene_list,
    }, f)

print(f"\nAll results saved to {OUTPUT_DIR}/")
print("Done!")
