#!/bin/bash
# ============================================================
# CSB10 Final: GEARS vs Additive Baseline Pipeline
# ============================================================
# Run from the CSB10_Final/ directory:
#   bash run_pipeline.sh
#
# Prerequisites (install once):
#   pip install torch torchvision
#   pip install torch-geometric
#   pip install scanpy anndata
#   pip install matplotlib seaborn scikit-learn tqdm pandas scipy
#
# Note on torch-geometric: installation depends on your PyTorch
# and CUDA version. See https://pytorch-geometric.readthedocs.io
# For CPU-only (Mac/laptop):
#   pip install torch-scatter torch-sparse torch-cluster \
#       -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)")+cpu.html
#   pip install torch-geometric
#
# First run downloads ~150 MB from Harvard Dataverse (needs internet).
# ============================================================

set -e
cd "$(dirname "$0")"

echo "============================================"
echo "CSB10 Final: Combinatorial Perturbation"
echo "            Prediction Pipeline"
echo "============================================"

# Step 1: Data loading + additive baseline
# First run: ~10-20 min (downloads + processes data)
# Cached:    ~2 min
echo ""
echo "[1/3] Loading data & computing additive baseline..."
python 01_data_and_baseline.py

# Step 2: GEARS training + evaluation
# CPU:  ~30-60 min (20 epochs)
# GPU:  ~5-10 min
# MPS:  ~15-20 min
echo ""
echo "[2/3] Training GEARS (20 epochs)..."
python 02_gears_training.py

# Step 3: Generate figures
# ~1 min
echo ""
echo "[3/3] Generating analysis figures..."
python 03_analysis.py

echo ""
echo "============================================"
echo "Done! Figures saved to results/"
echo "============================================"
echo ""
echo "Output files:"
echo "  results/fig1_training_curves.png"
echo "  results/fig2_tsne_embeddings.png"
echo "  results/fig3_subgroup_comparison.png"
echo "  results/fig4_per_pert_scatter.png"
echo "  results/fig5_example_perturbation.png"
echo "  results/model_comparison.csv"
echo "  results/training_history.csv"
echo "  results/additive_baseline_results.csv"
