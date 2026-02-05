# Parameter-Efficient Transformer Embeddings (PETE)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2505.02266-b31b1b.svg)](https://arxiv.org/abs/2505.02266) 

This repository contains the official implementation for the paper **"Parameter-Efficient Transformer Embedding"** by Henry Ndubuaku and Mouad Talhi.

## Overview

Traditional embedding layers in Transformer models often constitute the largest portion of parameters, scaling with vocabulary size without a proportional increase in performance. This project introduces PETE, a novel approach where token embeddings are generated deterministically using polynomial basis functions (Fourier, Chebyshev, Legendre, Laguerre, Hermite) applied to normalized token IDs, followed by a lightweight MLP.

This method significantly reduces the parameter count compared to standard learned embeddings, leading to faster training times and competitive performance, especially on sentence similarity tasks. The core polynomial expansions are implemented using efficient custom C++/CUDA kernels.

## Key Features

*   **Parameter Efficiency:** Replaces large learned embedding tables with deterministic polynomial expansions and a small MLP, drastically reducing parameters.
*   **Multiple Polynomial Bases:** Supports Fourier (default), Chebyshev, Legendre, Laguerre, and Hermite expansions. (Note: Currently, the code seems hardcoded to Fourier in `src/pete.py`, but the kernels exist).
*   **Custom Kernels:** High-performance C++/CUDA kernels for polynomial basis calculations.
*   **Competitive Performance:** Achieves strong results on benchmarks like STS-B, outperforming comparable small models.
*   **Faster Training:** Reduced parameter count and efficient kernels lead to quicker training cycles.

## Project Structure

```
.
├── polynomial_embeddings/ # C++/CUDA kernels for polynomial expansions
│   ├── *.cpp
│   ├── *.cu
│   └── *.h
├── src/                   # Python source code for model, training, data handling
│   ├── pete.py            # Main PETE model definition
│   ├── trainer.py         # Training and evaluation loops
│   ├── embedder.py        # Embedding wrapper and utility functions
│   ├── benchmark.py       # Evaluation functions
│   ├── data.py            # Data loading and processing
│   └── ...
├── paper/                 # LaTeX source for the paper
│   └── main.tex
├── environment.yml        # Conda environment specification
├── setup.py               # Setup for polynomial_embeddings package
└── README.md              # This file
```

## Installation

### Prerequisites

*   A Linux environment (tested on Ubuntu).
*   NVIDIA GPU with CUDA support.
*   `nvcc` (NVIDIA CUDA Compiler): Verify with `nvcc --version`. Install via package manager (e.g., `sudo apt install nvidia-cuda-toolkit`) or Conda.
*   `g++`: Verify with `which g++`. Install with `sudo apt install build-essential`.
*   Conda or Miniconda.

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HMUNACHI/pete.git 
    cd pete
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate pete_env
    ```

3.  **Install CUDA toolkit within the environment (adjust version if needed):**
    ```bash
    conda install -c nvidia/label/cuda-11.7.0 cuda-toolkit=11.7 cuda-nvcc=11.7
    # Or for newer CUDA versions:
    # conda install cuda -c nvidia
    ```

4.  **Compile and install the custom polynomial embedding kernels:**
    ```bash
    cd polynomial_embeddings
    pip install .
    cd ..
    ```

## Usage

### Training

To train a model using the default configuration (Fourier embeddings):

```bash
python src/trainer.py --experiment_name=pete_fourier_default
```

Check `src/trainer.py` (or potentially a separate script if arguments are added later) for command-line arguments to customize:

*   Model dimensions (`d_model`)
*   Number of layers (`num_hidden_layers`)
*   Number of attention heads (`num_attention_heads`)
*   Epochs, learning rate, batch size, etc.
*   Choice of polynomial embedding (Requires code modification in `src/pete.py` `PolynomialBlock` currently).

### Evaluation

The training script automatically runs evaluations on validation sets (like STS-B) during and after training. Results are logged to TensorBoard (`runs/`) and printed to the console. The best model weights are saved in the `weights/` directory.

### Reprpducing key ablation
```bash
  screen -S pete

  # 1_256                                                              
  python main.py --batch-size 512 --num-epochs 10 --d-model 256 && \
  python main.py --batch-size 512 --num-epochs 10 --d-model 256
  --permute-tokens && \
  python main.py --batch-size 512 --num-epochs 10 --d-model 256
  --random-embeddings && \

  # 1_512
  python main.py --batch-size 512 --num-epochs 10 --d-model 512 && \
  python main.py --batch-size 512 --num-epochs 10 --d-model 512
  --permute-tokens && \
  python main.py --batch-size 512 --num-epochs 10 --d-model 512
  --random-embeddings && \

  # 2_256
  python main.py --batch-size 512 --num-epochs 10 --num-hidden-layers 2
   --d-model 256 && \
  python main.py --batch-size 512 --num-epochs 10 --num-hidden-layers 2
   --d-model 256 --permute-tokens && \
  python main.py --batch-size 512 --num-epochs 10 --num-hidden-layers 2
   --d-model 256 --random-embeddings

  # Detach: press Ctrl+A, then D

  # Reattach later
  screen -r pete
```

### Using Different Polynomial Embeddings

Currently, the `PolynomialBlock` in `src/pete.py` is hardcoded to use `polynomial_embeddings.fourier`. To use other bases (Chebyshev, Legendre, Laguerre, Hermite), you would need to modify this line:

```python
# In src/pete.py -> PolynomialBlock.forward
# Change this line to use a different kernel:
embeddings = polynomial_embeddings.fourier( # Change 'fourier' to 'chebyshev', 'legendre', etc.
    input_ids, self.max_seq_len, self.d_model
)
```

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{ndubuaku2024pete,
  title={Parameter-Efficient Transformer Embedding},
  author={Ndubuaku, Henry and Talhi, Mouad},
  journal={arXiv preprint arXiv:2505.02266},
  year={2025}
}
```

*(Please update the BibTeX entry and arXiv link/badge when the paper is available.)*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (assuming MIT, add a LICENSE file if one doesn't exist).