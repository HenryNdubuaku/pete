# Parameter-Efficient Transformer Embeddings (PETE)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2505.02266-b31b1b.svg)](https://arxiv.org/abs/2505.02266) 

This repository contains the official implementation for the paper **"Parameter-Efficient Transformer Embedding"** by Henry Ndubuaku and Mouad Talhi.

## Overview

Traditional embedding layers in Transformer models often constitute the largest portion of parameters, scaling with vocabulary size without a proportional increase in performance. This project introduces PETE, a novel approach where token embeddings are generated deterministically using polynomial basis functions (Fourier, Chebyshev, Legendre, Laguerre, Hermite) applied to normalized token IDs, followed by a lightweight MLP.

This method significantly reduces the parameter count compared to standard learned embeddings, leading to faster training times and competitive performance, especially on sentence similarity tasks.

## Key Features

*   **Parameter Efficiency:** Replaces large learned embedding tables with deterministic Fourier embeddings and a small MLP, drastically reducing parameters.
*   **Pure PyTorch:** No custom CUDA kernels required - runs on any PyTorch-supported device.
*   **Competitive Performance:** Achieves strong results on benchmarks like STS-B, outperforming comparable small models.
*   **Faster Training:** Reduced parameter count leads to quicker training cycles.

## Project Structure

```
.
├── src/
│   ├── pete.py                  # Main PETE model (Fourier embeddings)
│   ├── transformer.py           # Baseline transformer model
│   ├── polynomial_embeddings.py # Triton kernels for polynomial bases
│   ├── trainer.py               # Training loops
│   ├── embedder.py              # Contrastive learning wrapper
│   ├── benchmark.py             # GLUE evaluation functions
│   ├── data.py                  # Data loading and processing
│   └── utils.py                 # Utilities
├── main.py                      # Entry point for training
├── requirements.txt             # Python dependencies
└── README.md
```

## Installation

```bash
git clone https://github.com/HMUNACHI/pete.git
cd pete
pip install -r requirements.txt
```

## Usage

### Training

```bash
python main.py --batch-size 512 --num-epochs 10
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--d-model` | 128 | Model dimension |
| `--num-hidden-layers` | 1 | Number of transformer layers |
| `--batch-size` | 256 | Batch size |
| `--num-epochs` | 5 | Number of epochs |
| `--learning-rate` | 1e-5 | Learning rate |
| `--include-baseline` | False | Also train standard transformer |
| `--permute-tokens` | False | Ablation: randomize token IDs before embedding |
| `--random-embeddings` | False | Ablation: use Random Fourier Features |
| `--index-mode` | raw | Index mapping: `raw`, `normalized`, or `scaled` |
| `--index-scale` | 1.0 | Scale factor for `--index-mode=scaled` |
| `--rff-sigma` | None | Frequency scale for Random Fourier Features |

### Evaluation

The training script automatically runs evaluations on validation sets (like STS-B) during and after training. Results are logged to TensorBoard (`runs/`) and printed to the console. The best model weights are saved in the `weights/` directory.

### Reproducing key ablations
```bash
# Transformer baselines
python main.py --batch-size 512 --num-epochs 10 --d-model 256 --include-baseline && \
python main.py --batch-size 512 --num-epochs 10 --d-model 512 --include-baseline && \
python main.py --batch-size 512 --num-epochs 10 --num-hidden-layers 2 --d-model 256 --include-baseline

# 2_256 ablations
python main.py --batch-size 512 --num-epochs 10 --num-hidden-layers 2 --d-model 256
python main.py --batch-size 512 --num-epochs 10 --num-hidden-layers 2 --d-model 256 --permute-tokens && \
python main.py --batch-size 512 --num-epochs 10 --num-hidden-layers 2 --d-model 256 --random-embeddings && \
python main.py --batch-size 512 --num-epochs 10 --num-hidden-layers 2 --d-model 256 --index-mode normalized && \
python main.py --batch-size 512 --num-epochs 10 --num-hidden-layers 2 --d-model 256 --index-mode scaled --index-scale 0.001
```

### Exploring embedding similarity

After training, compare PETE's learned Fourier+MLP embeddings with traditional transformer embeddings:

```bash
# Compare embeddings for all configs
python explore_similarity.py --config 1_256 && \
python explore_similarity.py --config 1_512 && \
python explore_similarity.py --config 2_256
```

This computes cosine similarity, angular distance, and MSE between PETE and transformer embedding layers to analyze whether PETE learns to approximate traditional embeddings.

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{ndubuaku2024pete,
  title={Parameter-Efficient Transformer Embedding},
  author={Ndubuaku et. al},
  journal={arXiv preprint arXiv:2505.02266},
  year={2025}
}
```

*(Please update the BibTeX entry and arXiv link/badge when the paper is available.)*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (assuming MIT, add a LICENSE file if one doesn't exist).