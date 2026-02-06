"""
Exploration script to compare PETE's learned Fourier+MLP embeddings
with traditional transformer token embeddings.

Computes cosine similarity, angular distance, and MSE between:
- Transformer: embedding layer output
- PETE: Fourier expansion + MLP output
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from src.data import GlueDatasetLoader
from src.pete import PETE
from src.transformer import Transformer


def load_models(
    config: str,
    vocab_size: int,
    max_seq_len: int,
    index_mode: str = "raw",
    index_scale: float = 1.0,
    rff_sigma: float = None,
    permute_tokens: bool = False,
    random_embeddings: bool = False,
):
    """Load PETE and Transformer models from saved weights."""
    parts = config.split("_")
    num_layers = int(parts[0])
    d_model = int(parts[1])

    pete = PETE(
        vocab_size=vocab_size,
        d_model=d_model,
        num_hidden_layers=num_layers,
        num_attention_heads=num_layers,
        max_seq_len=max_seq_len,
        index_mode=index_mode,
        index_scale=index_scale,
        rff_sigma=rff_sigma,
        permute_tokens=permute_tokens,
        random_embeddings=random_embeddings,
    )

    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_hidden_layers=num_layers,
        num_attention_heads=num_layers,
        max_position_embeddings=max_seq_len,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        intermediate_size=0,
    )

    # Load weights
    pete_path = f"weights/pete_{config}.pt"
    transformer_path = f"weights/transformer_{config}.pt"

    if os.path.exists(pete_path):
        state_dict = torch.load(pete_path, map_location="cpu", weights_only=False)
        # Handle weights saved from Embedder wrapper (prefixed with "model.")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_key = k[6:]  # Remove "model." prefix
                new_state_dict[new_key] = v
            elif k in ("temperature", "classifier.weight", "classifier.bias"):
                # Skip Embedder-specific keys
                continue
            else:
                new_state_dict[k] = v
        # Handle old key names (harmonics -> inv_freq)
        if "expansion.harmonics" in new_state_dict and "expansion.inv_freq" not in new_state_dict:
            new_state_dict["expansion.inv_freq"] = new_state_dict.pop("expansion.harmonics")
        pete.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded PETE weights from {pete_path}")
    else:
        print(f"Warning: {pete_path} not found, using random weights")

    if os.path.exists(transformer_path):
        state_dict = torch.load(transformer_path, map_location="cpu", weights_only=False)
        # Handle weights saved from Embedder wrapper (prefixed with "model.")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_key = k[6:]  # Remove "model." prefix
                new_state_dict[new_key] = v
            elif k in ("temperature", "classifier.weight", "classifier.bias"):
                continue
            else:
                new_state_dict[k] = v
        transformer.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded Transformer weights from {transformer_path}")
    else:
        print(f"Warning: {transformer_path} not found, using random weights")

    return pete, transformer


def get_pete_embeddings(model: PETE, input_ids: torch.Tensor) -> torch.Tensor:
    """Get PETE embeddings: Fourier expansion + MLP (before transformer layers)."""
    with torch.no_grad():
        x_polynomials = model.expansion(input_ids)
        x = model.norm(model.mlp(x_polynomials) + x_polynomials)
    return x


def get_transformer_embeddings(model: Transformer, input_ids: torch.Tensor) -> torch.Tensor:
    """Get Transformer embeddings: token embedding layer output (before transformer layers)."""
    with torch.no_grad():
        x = model.embeddings(input_ids)
        x = model.dropout(model.ln(x))
    return x


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean cosine similarity between two embedding tensors."""
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])

    a_norm = F.normalize(a_flat, p=2, dim=-1)
    b_norm = F.normalize(b_flat, p=2, dim=-1)

    cos_sim = (a_norm * b_norm).sum(dim=-1)
    return cos_sim.mean().item()


def angular_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean angular distance (in degrees) between two embedding tensors."""
    cos_sim = cosine_similarity_matrix(a, b)
    # Clamp to avoid numerical issues with arccos
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    angles = torch.acos(cos_sim) * (180.0 / np.pi)
    return angles.mean().item()


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute element-wise cosine similarity."""
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])

    a_norm = F.normalize(a_flat, p=2, dim=-1)
    b_norm = F.normalize(b_flat, p=2, dim=-1)

    return (a_norm * b_norm).sum(dim=-1)


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean squared error between two embedding tensors."""
    return F.mse_loss(a, b).item()


def analyze_embeddings(pete: PETE, transformer: Transformer, dataloader, device: torch.device):
    """Analyze embedding similarities across a dataset."""
    pete.to(device).eval()
    transformer.to(device).eval()

    all_cos_sims = []
    all_angular_dists = []
    all_mses = []

    for batch in dataloader:
        input_ids = batch[0].to(device)

        pete_emb = get_pete_embeddings(pete, input_ids)
        transformer_emb = get_transformer_embeddings(transformer, input_ids)

        cos_sim = cosine_similarity(pete_emb, transformer_emb)
        ang_dist = angular_distance(pete_emb, transformer_emb)
        mse_val = mse(pete_emb, transformer_emb)

        all_cos_sims.append(cos_sim)
        all_angular_dists.append(ang_dist)
        all_mses.append(mse_val)

    return {
        "cosine_similarity": np.mean(all_cos_sims),
        "angular_distance_deg": np.mean(all_angular_dists),
        "mse": np.mean(all_mses),
    }


def analyze_per_token(pete: PETE, transformer: Transformer, vocab_size: int, device: torch.device):
    """Analyze embeddings for each token in the vocabulary."""
    pete.to(device).eval()
    transformer.to(device).eval()

    # Create input with all token IDs
    all_tokens = torch.arange(vocab_size, device=device).unsqueeze(0)  # (1, vocab_size)

    pete_emb = get_pete_embeddings(pete, all_tokens)  # (1, vocab_size, d_model)
    transformer_emb = get_transformer_embeddings(transformer, all_tokens)  # (1, vocab_size, d_model)

    # Compute per-token metrics
    pete_emb = pete_emb.squeeze(0)  # (vocab_size, d_model)
    transformer_emb = transformer_emb.squeeze(0)  # (vocab_size, d_model)

    # Cosine similarity per token
    pete_norm = F.normalize(pete_emb, p=2, dim=-1)
    transformer_norm = F.normalize(transformer_emb, p=2, dim=-1)
    per_token_cos_sim = (pete_norm * transformer_norm).sum(dim=-1)

    # MSE per token
    per_token_mse = ((pete_emb - transformer_emb) ** 2).mean(dim=-1)

    return {
        "mean_cosine_similarity": per_token_cos_sim.mean().item(),
        "std_cosine_similarity": per_token_cos_sim.std().item(),
        "min_cosine_similarity": per_token_cos_sim.min().item(),
        "max_cosine_similarity": per_token_cos_sim.max().item(),
        "mean_mse": per_token_mse.mean().item(),
        "std_mse": per_token_mse.std().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Explore PETE vs Transformer embeddings")
    parser.add_argument("--config", type=str, default="1_128", help="Model config (e.g., 1_128, 2_256)")
    parser.add_argument("--vocab-size", type=int, default=30552, help="Vocabulary size")
    parser.add_argument("--max-seq-len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--index-mode", type=str, default="raw", choices=["raw", "normalized", "scaled"],
                        help="Index mapping mode")
    parser.add_argument("--index-scale", type=float, default=1.0, help="Scale for index-mode=scaled")
    parser.add_argument("--rff-sigma", type=float, default=None, help="RFF frequency scale")
    parser.add_argument("--permute-tokens", action="store_true", help="Use permuted token IDs")
    parser.add_argument("--random-embeddings", action="store_true", help="Use Random Fourier Features")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    pete, transformer = load_models(
        args.config,
        args.vocab_size,
        args.max_seq_len,
        index_mode=args.index_mode,
        index_scale=args.index_scale,
        rff_sigma=args.rff_sigma,
        permute_tokens=args.permute_tokens,
        random_embeddings=args.random_embeddings,
    )

    # Load tokenizer and data
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    data = GlueDatasetLoader(
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        batch_size=args.batch_size,
        dataset_names=["stsb"],
    )

    print(f"\n{'='*60}")
    print(f"Comparing PETE vs Transformer embeddings (config: {args.config})")
    print(f"{'='*60}")

    # Analyze on STS-B validation set
    print("\n--- STS-B Validation Set Analysis ---")
    stsb_results = analyze_embeddings(
        pete, transformer,
        data.data_loaders["stsb"]["validation"],
        device
    )
    for metric, value in stsb_results.items():
        print(f"  {metric}: {value:.6f}")

    # Analyze per-token across vocabulary
    print("\n--- Per-Token Vocabulary Analysis ---")
    vocab_results = analyze_per_token(pete, transformer, args.vocab_size, device)
    for metric, value in vocab_results.items():
        print(f"  {metric}: {value:.6f}")

    print(f"\n{'='*60}")
    print("Interpretation:")
    print("  - Cosine similarity close to 1.0 = similar representations")
    print("  - Angular distance close to 0° = similar directions")
    print("  - Low MSE = similar magnitudes and directions")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
