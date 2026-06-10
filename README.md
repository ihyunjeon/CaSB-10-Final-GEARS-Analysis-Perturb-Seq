# CSB10 Final Project: Predicting Combinatorial Perturbation Outcomes

**Team:** Anson Ting, Ihyun Jeon, Jun Han

## Overview

We predict two-gene combinatorial CRISPRa perturbation outcomes from single-gene perturbation data using the Norman et al. 2019 dataset. We compare an additive baseline against GEARS (Roohani et al. 2023) and analyze learned perturbation embeddings via t-SNE.

## Dataset

- **Source:** Norman et al. 2019 (loaded via GEARS PertData from Harvard Dataverse)
- ~111K K562 cells, ~5,000 genes (after GEARS preprocessing)
- 105 single-gene perturbations, 131 two-gene combinatorial perturbations
- **Split:** GEARS simulation split (holds out unseen 2-gene combos as test set)

## Project Structure

```
CSB10_Final/
├── README.md
├── requirements.txt
├── run_pipeline.sh              # One-command runner
├── 01_data_and_baseline.py      # Load data via GEARS, compute additive baseline
├── 02_gears_training.py         # Train GEARS with per-epoch tracking
├── 03_analysis.py               # Generate 5 presentation figures
├── GEARS/                       # GEARS source (from author)
│   ├── gears/
│   └── demo/
├── results/                     # Output figures, metrics, model
└── data/                        # (auto-downloaded) GEARS-processed data
```

## Setup

```bash
pip install torch torchvision torch-geometric
pip install scanpy anndata matplotlib seaborn scikit-learn tqdm
```

See `run_pipeline.sh` for torch-geometric installation notes.

## Pipeline

```bash
bash run_pipeline.sh
```

Or step by step:

1. `python 01_data_and_baseline.py` — loads Norman data via GEARS PertData, computes additive baseline on simulation split test set
2. `python 02_gears_training.py` — trains GEARS (20 epochs), evaluates on same test set, extracts embeddings
3. `python 03_analysis.py` — generates 5 figures comparing models

## Output Figures

| Figure | Description |
|--------|-------------|
| `fig1_training_curves.png` | Training loss and validation MSE per epoch |
| `fig2_tsne_embeddings.png` | t-SNE of learned perturbation embeddings (post-GNN) |
| `fig3_subgroup_comparison.png` | Additive vs GEARS Pearson r by subgroup (seen0/1/2) |
| `fig4_per_pert_scatter.png` | Per-combo scatter: additive r vs GEARS r |
| `fig5_example_perturbation.png` | Box plot of actual expression + model predictions for best example combo |

## Key Design Decisions

- **Single data pipeline:** Both models use GEARS's `PertData` with the same simulation split, ensuring apples-to-apples comparison on identical test combos and DE gene sets.
- **Subgroup analysis:** GEARS's simulation split categorizes test combos by how many component genes were seen in training combos (seen0/seen1/seen2), enabling difficulty-stratified evaluation.
- **Post-GNN embeddings:** t-SNE uses perturbation embeddings after gene ontology GNN augmentation, capturing learned biological relationships.

## References

- Norman et al. (2019). *Exploring genetic interaction manifolds constructed from rich single-cell phenotypes.* Science.
- Roohani, Huang, Leskovec (2023). *Predicting transcriptional outcomes of novel multigene perturbations with GEARS.* Nature Biotechnology.
