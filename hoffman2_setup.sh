#!/bin/bash
# ============================================================
# One-time Hoffman2 environment setup
# Run this INTERACTIVELY (not via qsub) after SSH-ing in.
# ============================================================

echo "Setting up GEARS conda environment on Hoffman2..."

# Load conda
module load anaconda3

# Create environment with PyTorch + CUDA
conda create -n gears_env python=3.10 -y
source activate gears_env

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-geometric

# Install remaining dependencies
pip install scanpy anndata matplotlib seaborn scikit-learn tqdm pandas scipy

echo ""
echo "Done! Test with:"
echo "  source activate gears_env"
echo "  python -c \"import torch; print(torch.cuda.is_available())\""
