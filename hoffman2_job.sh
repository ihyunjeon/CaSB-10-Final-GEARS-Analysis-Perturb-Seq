#!/bin/bash
# ============================================================
# Hoffman2 GPU Job Script for GEARS Training
# Submit with:  qsub hoffman2_job.sh
# ============================================================

#$ -cwd
#$ -o joblog.$JOB_ID
#$ -j y
#$ -N gears_train
#$ -l h_data=16G,h_rt=4:00:00,gpu,A100
#$ -pe shared 4
#$ -M $USER
#$ -m bea

echo "================================================"
echo "Job ID: $JOB_ID"
echo "Host:   $(hostname)"
echo "Date:   $(date)"
echo "================================================"

# Initialize module system and conda on compute node
. /u/local/Modules/default/init/modules.sh
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate gears_env

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Run pipeline
echo ""
echo "[1/3] Data loading & additive baseline..."
python 01_data_and_baseline.py

echo ""
echo "[2/3] Training GEARS (20 epochs on GPU)..."
python 02_gears_training.py

echo ""
echo "[3/3] Generating analysis figures..."
python 03_analysis.py

echo ""
echo "================================================"
echo "Done! $(date)"
echo "================================================"
