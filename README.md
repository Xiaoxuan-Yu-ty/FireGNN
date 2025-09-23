# FireGNN: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification

## Overview

FireGNN is a framework for fuzzy rule-enhanced graph neural networks that combines trainable fuzzy rules with auxiliary tasks for improved performance on medical image classification. The system includes baseline models, fuzzy-enhanced models, and auxiliary task models.

### Paper Accepted at NeurIPS 2025 Workshop[NPGML;BranBodyFM], San Diego, USA 🎉

## Architecture
<img width="625" height="351" alt="image" src="https://github.com/user-attachments/assets/3eaa5f0d-8665-4f02-96db-7155926cb089" />

## Demo

https://github.com/user-attachments/assets/6efb0935-37c8-4a46-97a7-2fa1d88a98c0


## Project Structure
```
FireGNN/
├── data_processing/     # Graph construction and data preprocessing
├── models/             # Baseline GNN models (GCN, GAT, GIN)
├── fuzzy_models/       # Fuzzy rule-enhanced models
├── auxiliary_models/   # Models with auxiliary tasks
├── utils/              # Utility functions and helpers
├── datasets/           # Pre-built graph datasets
├── results/            # Output results and visualizations
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Key Features

- **Graph Construction**: Converts MedMNIST and MorphoMNIST datasets to k-NN graphs
- **Baseline Models**: Standard GNN implementations (GCN, GAT, GIN)
- **Fuzzy Models**: GNNs enhanced with trainable fuzzy rules
- **Auxiliary Models**: GNNs with auxiliary tasks for improved learning
- **Multiple Datasets**: Support for all MedMNIST variants and MorphoMNIST
- **Comprehensive Evaluation**: Cross-validation, multiple metrics, and visualizations

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Setup

### 1. Download MedMNIST Datasets

The MedMNIST datasets will be automatically downloaded when you first run the graph construction scripts. However, you can also download them manually:

```bash
# Install MedMNIST
pip install medmnist

# The datasets will be downloaded automatically when running:
python data_processing/build_graphs.py --dataset organcmnist
```

### 2. Download MorphoMNIST Dataset

For MorphoMNIST, you need to download it manually:

```bash
# Create directory for MorphoMNIST
mkdir -p datasets/morpho-mnist

# Download MorphoMNIST (you'll need to download from the official source)
# Place the files in datasets/morpho-mnist/
# Structure should be:
# datasets/morpho-mnist/
# ├── plain/
# │   ├── train-images-idx3-ubyte.gz
# │   ├── train-labels-idx1-ubyte.gz
# │   ├── t10k-images-idx3-ubyte.gz
# │   └── t10k-labels-idx1-ubyte.gz
# ├── thic/
# ├── thin/
# └── frac/
```

### 3. Build Graphs from Datasets

```bash
# Build graphs for MedMNIST datasets
python data_processing/build_graphs.py --dataset organcmnist
python data_processing/build_graphs.py --dataset bloodmnist
python data_processing/build_graphs.py --dataset tissuemnist
python data_processing/build_graphs.py --dataset organamnist
python data_processing/build_graphs.py --dataset organsmnist

# Build graph for MorphoMNIST
python data_processing/build_morpho_graphs.py --variant plain
python data_processing/build_morpho_graphs.py --variant thic
python data_processing/build_morpho_graphs.py --variant thin
python data_processing/build_morpho_graphs.py --variant frac
```

## Usage

### 1. Baseline Models

Train baseline GNN models:

```bash
python models/train_baseline.py --model gcn --dataset organcmnist
python models/train_baseline.py --model gat --dataset organcmnist
python models/train_baseline.py --model gin --dataset organcmnist
```

### 2. Fuzzy Models

Train fuzzy rule-enhanced models:

```bash
python fuzzy_models/train_fuzzy.py --model gcn --dataset organcmnist
python fuzzy_models/train_fuzzy.py --model gat --dataset organcmnist
python fuzzy_models/train_fuzzy.py --model gin --dataset organcmnist
```

### 3. Auxiliary Models

Train models with auxiliary tasks:

```bash
python auxiliary_models/train_auxiliary.py --model gcn --dataset organcmnist
python auxiliary_models/train_auxiliary.py --model gat --dataset organcmnist
python auxiliary_models/train_auxiliary.py --model gin --dataset organcmnist
```

### 4. Run Complete Pipeline

```bash
# Run all models for a dataset
python run_pipeline.py --dataset organcmnist --train_baselines --train_fuzzy --train_auxiliary

# Run with specific models
python run_pipeline.py --dataset organcmnist --models gcn gat --train_baselines --train_fuzzy
```

## Model Types

### Baseline Models
- **GCN**: Graph Convolutional Network
- **GAT**: Graph Attention Network  
- **GIN**: Graph Isomorphism Network

### Fuzzy Models
- **Fuzzy-GCN**: GCN with trainable fuzzy rules
- **Fuzzy-GAT**: GAT with trainable fuzzy rules
- **Fuzzy-GIN**: GIN with trainable fuzzy rules

### Auxiliary Models
- **Aux-GCN**: GCN with auxiliary tasks
- **Aux-GAT**: GAT with auxiliary tasks
- **Aux-GIN**: GIN with auxiliary tasks

## Fuzzy Rules

The fuzzy models use trainable Gaussian membership functions:

- **Rule Centers**: Learnable centers for each fuzzy rule
- **Rule Widths**: Learnable standard deviations
- **Rule Weights**: Learnable importance weights
- **Rule Integration**: Fuzzy rules integrated with GNN embeddings

<img width="635" height="147" alt="image" src="https://github.com/user-attachments/assets/b59ea05d-c0d4-45bf-8ae9-0b941b933e2e" />

# Demo: Theta Analysis

https://github.com/user-attachments/assets/21f74319-62d0-40d1-a87b-2934650889fe

## Auxiliary Tasks

The auxiliary models include additional learning objectives:

- **Similarity Prediction**: Predict node similarity scores
- **Homophily Prediction**: Predict local homophily measures
- **Entropy Prediction**: Predict neighborhood entropy
- **Multi-task Learning**: Combine main classification with auxiliary tasks

## Datasets

### MedMNIST Variants
- **OrganCMNIST**: Medical organ classification (11 classes)
- **BloodMNIST**: Blood cell classification (8 classes)
- **TissueMNIST**: Tissue type classification (8 classes)
- **OrganAMNIST**: Organ classification variant (11 classes)
- **OrganSMNIST**: Organ classification variant (11 classes)

### MorphoMNIST Variants
- **Plain**: Standard morphological digits
- **Thic**: Thick morphological digits
- **Thin**: Thin morphological digits
- **Frac**: Fractal morphological digits

## Results

Results are saved in the `results/` directory with:
- Training curves and metrics
- Cross-validation results
- Model comparisons
- Fuzzy rule visualizations
- t-SNE embeddings
- Confusion matrices

## Configuration

Set environment variables for reproducibility:
```bash
export CUDA_VISIBLE_DEVICES=0  # GPU device
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and build graphs
python data_processing/build_graphs.py --dataset organcmnist

# 3. Run quick test
python quick_start.py

# 4. Run full pipeline
python run_pipeline.py --dataset organcmnist --train_baselines --train_fuzzy --train_auxiliary
```

## Citation

If you use this code in your research, please cite:
```
@misc{sengupta2025firegnnneurosymbolicgraphneural,
      title={FireGNN: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification}, 
      author={Prajit Sengupta and Islem Rekik},
      year={2025},
      eprint={2509.10510},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2509.10510}, 
}

@inproceedings{
anonymous2025firegnn,
title={Fire{GNN}: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification},
author={Anonymous},
booktitle={Foundation Models for the Brain and Body Workshop {\textendash} NeurIPS 2025},
year={2025},
url={https://openreview.net/forum?id=y3dhKusYFx}
}

@inproceedings{
sengupta2025firegnn,
title={Fire{GNN}: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification},
author={Prajit Sengupta and Islem Rekik},
booktitle={New Perspectives in Graph Machine Learning},
year={2025},
url={https://openreview.net/forum?id=jCwmNcGNCC}
}
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Dataset download fails**: Check internet connection and try again
3. **Graph construction slow**: Use smaller k values for k-NN
4. **Fuzzy rules not converging**: Adjust learning rate or rule initialization

### Performance Tips

1. Use GPU acceleration when available
2. Adjust number of fuzzy rules based on dataset size
3. Tune auxiliary task weights for optimal performance
4. Use early stopping to prevent overfitting 
