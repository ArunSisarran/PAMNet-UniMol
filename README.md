# PAMNet-UniMol

A fusion model combining **PAMNet** (Physics-aware Multiplex Graph Neural Network) and **UniMol** (Universal Molecular Foundation Model) for enhanced molecular property prediction.

## Overview

PAMNet-UniMol leverages the complementary strengths of two state-of-the-art molecular representation approaches:

- **PAMNet**: Captures 3D geometric and physical properties through multiplex graph neural networks with dual-layer message passing (local + global interactions)
- **UniMol**: Provides powerful pre-trained molecular embeddings learned from millions of molecular structures

The fusion architecture uses **cross-attention** and **gating mechanisms** to intelligently combine both representations, achieving significant improvements over either model alone.

## Results

| Dataset | Metric | Improvement over PAMNet |
|---------|--------|------------------------|
| QM9     | MAE    | **33%** reduction      |
| ADMET   | MAE    | **30%** reduction      |

## Architecture

```
┌─ PAMNet (3D Geometric) ──────────┐
│   - Local message passing        │
│   - Global message passing       │
│   - Bessel/Spherical basis       │
│                                  ↓
│                         Cross-Attention
│                         Fusion Block
│                                  ↑
└── UniMol (Pre-trained) ─────────┘
    - 512-dim embeddings           │
    - Foundation model features    ↓
                            Gated Ensemble
                                  │
                                  ↓
                           Final Prediction
```

The gating mechanism learns to optimally blend PAMNet predictions with fusion predictions, maintaining robustness while leveraging the benefits of both models.

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.4.0+
- PyTorch Geometric
- RDKit
- UniMol

```bash
# Clone the repository
git clone https://github.com/yourusername/PAMNet-UniMol.git
cd PAMNet-UniMol

# Install dependencies
pip install numpy scikit-learn scipy sympy rdkit-pypi
pip install torch torchvision
pip install torch-geometric torch-scatter torch-sparse torch-cluster

# Install UniMol (follow UniMol installation instructions)
# Install warmup scheduler
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
```

## Usage

### 1. Precompute UniMol Embeddings

Before training, precompute UniMol embeddings for your dataset:

```bash
cd pamnet+unimol/

# For ADMET datasets
python precompute_embeddings.py \
  --dataset_type ADMET \
  --dataset_name Caco2_Wang \
  --batch_size 256

# For QM9 dataset
python precompute_embeddings.py \
  --dataset_type QM9 \
  --target 7
```

### 2. Train the Fusion Model

```bash
# Train on ADMET dataset
python train_admet_fusion.py \
  --dataset Caco2_Wang \
  --task regression \
  --epochs 100 \
  --lr 1e-4 \
  --batch_size 16 \
  --fusion_dim 128 \
  --num_heads 4

# Train on QM9 dataset
python train_attention_fusion.py \
  --target 7 \
  --epochs 900 \
  --lr 1e-4
```

### 3. Inference

```python
from data_processing import DataProcessing
from attention_fusion_model import Attention_Fusion

# Initialize processor
processor = DataProcessing()

# Process SMILES to get embeddings and graph data
result = processor.data_process_pamnet_unimol(["CCO", "CC(=O)O"])

# Load trained model and run inference
model = Attention_Fusion(...)
model.load_state_dict(torch.load("checkpoint.pt"))
predictions = model(graph_data, embeddings)
```

## Datasets

The model supports multiple molecular property prediction datasets:

| Dataset | Task | Description |
|---------|------|-------------|
| **QM9** | Regression | 12 quantum mechanical properties (~130k molecules) |
| **ADMET** | Regression/Classification | Drug-like properties from TDC (Caco2, HIA, hERG, Solubility, etc.) |
| **PDBbind** | Regression | Protein-ligand binding affinity |
| **RNA-Puzzles** | Regression | RNA structure prediction (RMSD) |

## Project Structure

```
PAMNet-UniMol/
├── Physics-aware-Multiplex-GNN/    # PAMNet implementation
│   ├── models.py                   # PAMNet & PAMNet_s models
│   ├── layers/                     # Message passing layers
│   ├── datasets/                   # Dataset implementations
│   └── utils/                      # Utilities (EMA, metrics, etc.)
│
├── pamnet+unimol/                  # Fusion model
│   ├── attention_fusion_model.py   # Main fusion architecture
│   ├── train_attention_fusion.py   # QM9 training script
│   ├── train_admet_fusion.py       # ADMET training script
│   ├── precompute_embeddings.py    # UniMol embedding generation
│   └── data_processing.py          # SMILES to graph conversion
```

## Key Features

- **Cross-attention fusion**: Bidirectional attention between PAMNet and UniMol representations
- **Learnable gating**: Automatically balances between baseline PAMNet and fusion predictions
- **Transfer learning**: Supports freezing PAMNet for adaptation to new datasets
- **EMA smoothing**: Exponential moving average for improved generalization
- **Flexible training**: Supports both regression and classification tasks

## Configuration

Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fusion_dim` | 128 | Projection dimension for fusion |
| `num_heads` | 4 | Number of attention heads |
| `dropout` | 0.1 | Dropout rate |
| `lr` | 1e-4 | Learning rate |
| `warmup_epochs` | 5 | Learning rate warmup period |
| `patience` | 30 | Early stopping patience |

## Citation

If you use this work, please cite:

```bibtex
@article{pamnet2023,
  title={A universal framework for accurate and efficient geometric deep learning of molecular systems},
  journal={Nature Scientific Reports},
  year={2023}
}
```

## License

This project builds upon PAMNet and UniMol. Please refer to the original repositories for licensing information.

## Acknowledgments

- [PAMNet](https://github.com/XieResearchGroup/Physics-aware-Multiplex-GNN) - Physics-aware Multiplex Graph Neural Network
- [UniMol](https://github.com/dptech-corp/Uni-Mol) - Universal Molecular Representation Learning
