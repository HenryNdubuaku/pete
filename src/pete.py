import math
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from torch import nn

torch.manual_seed(0)
np.random.seed(0)


class PolynomialBlock(nn.Module):
    """
    Pure PyTorch implementation of PETE-style fixed Fourier features over token IDs.

    Transforms token IDs into dense embeddings using sinusoidal functions
    at exponentially spaced frequencies (like positional encoding), or
    a Random Fourier Features (RFF) ablation with random frequencies + phases.

    Supports ablations:
    - permute_tokens: Randomly permute token IDs before embedding (tests reliance on tokenizer ID structure)
    - random_embeddings: Use Random Fourier Features (random frequencies + phases) instead of deterministic inv_freq

    Also supports index mapping ablation:
    - index_mode:
        * "raw"        : x = p
        * "normalized" : x = 2*(p/(V-1)) - 1
        * "scaled"     : x = scale * p
    """

    def __init__(
        self,
        max_seq_len: int,
        d_model: int,
        vocab_size: int,
        permute_tokens: bool = False,
        random_embeddings: bool = False,
        seed: int = 42,
        base: float = 10000.0,
        index_mode: str = "raw",
        index_scale: float = 1.0,
        # Optional: control RFF frequency scale; if None, defaults to mean(inv_freq)
        rff_sigma: Optional[float] = None,
    ):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even (got {d_model})")

        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.permute_tokens = permute_tokens
        self.random_embeddings = random_embeddings
        self.base = base

        if index_mode not in ("raw", "normalized", "scaled"):
            raise ValueError(f"index_mode must be one of ['raw','normalized','scaled'] (got {index_mode})")
        self.index_mode = index_mode
        self.index_scale = float(index_scale)

        half = d_model // 2

        # Deterministic exponentially-spaced frequencies (PE-style)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)  # shape (half,)

        if random_embeddings:
            # Random Fourier Features: angles = x * w + b
            # Choose sigma in the same rough scale as deterministic inv_freq unless overridden.
            if rff_sigma is None:
                # Use mean inv_freq as a reasonable default scale for w.
                rff_sigma = float(inv_freq.mean().item())

            g = torch.Generator().manual_seed(seed)
            # Frequencies w ~ N(0, sigma^2)
            random_freqs = torch.randn(half, generator=g) * rff_sigma
            # Phases b ~ U(0, 2π)
            random_phases = torch.rand(half, generator=g) * (2.0 * math.pi)

            self.register_buffer("random_freqs", random_freqs)     # shape (half,)
            self.register_buffer("random_phases", random_phases)   # shape (half,)

        if permute_tokens:
            g = torch.Generator().manual_seed(seed)
            permutation = torch.randperm(vocab_size, generator=g)
            self.register_buffer("permutation", permutation)

    def _map_indices(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs to scalar x according to index_mode."""
        x = input_ids.float()

        if self.index_mode == "raw":
            # x = p
            pass
        elif self.index_mode == "scaled":
            # x = scale * p
            x = x * self.index_scale
        elif self.index_mode == "normalized":
            # x = 2*(p/(V-1)) - 1
            denom = float(self.vocab_size - 1) if self.vocab_size > 1 else 1.0
            x = 2.0 * (x / denom) - 1.0
        else:
            raise RuntimeError("unreachable")

        return x.unsqueeze(-1)  # (batch, seq, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Long tensor of shape (batch_size, seq_len)

        Returns:
            embeddings: Float tensor of shape (batch_size, seq_len, d_model)
        """
        if input_ids.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
            # Allow non-long ids but they must be integer-like.
            # If you always pass torch.long, you can delete this.
            input_ids = input_ids.long()

        if self.permute_tokens:
            input_ids = self.permutation[input_ids]

        x = self._map_indices(input_ids)  # (batch, seq, 1)

        if self.random_embeddings:
            # Random Fourier Features: sin(w*x + b), cos(w*x + b)
            w = self.random_freqs.to(x.device)        # (half,)
            b = self.random_phases.to(x.device)       # (half,)
            angles = x * w + b                        # broadcast -> (batch, seq, half)
        else:
            # Deterministic Fourier features with exponentially spaced frequencies
            w = self.inv_freq.to(x.device)            # (half,)
            angles = x * w                            # (batch, seq, half)

        sin_emb = torch.sin(angles)
        cos_emb = torch.cos(angles)
        embeddings = torch.cat([sin_emb, cos_emb], dim=-1)  # (batch, seq, d_model)

        return embeddings


class RotaryPositionEncoding(nn.Module):
    """
    Implements Rotary Position Encoding for attention mechanism.
    """

    def __init__(self, dim, max_seq_len, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_seq_len}"
            )

        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0)
        sin = emb.sin().unsqueeze(0)
        return cos, sin

    def apply_rotary_pos_emb(self, x, cos, sin):
        x_rot = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x_rot.unbind(-1)
        x = torch.stack([-x2, x1], dim=-1)
        x = x.reshape(*x.shape[:-2], -1)
        return (x * cos) + (x.roll(shifts=1, dims=-1) * sin)


class MLP(nn.Module):
    """
    Implements a decomposed linear layer with an intermediate activation.
    """
    def __init__(self, in_features, out_features, is_pete=False):
        super(MLP, self).__init__()
        if is_pete:
            intermidiate = in_features // 4
        else:
            intermidiate = in_features * 4
        self.w1 = nn.Linear(in_features, intermidiate*2)
        self.w2 = nn.Linear(intermidiate, out_features)

    def forward(self, x):
        x = self.w1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * nn.functional.gelu(gate)
        return self.w2(x)


class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Layer Normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, dim=-1) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=dim, keepdim=True) + self.eps)
        x_normalized = x / rms
        return self.weight * x_normalized


class Layer(nn.Module):
    """
    Implements a multi-head attention block with feed-forward network.
    """

    def __init__(
        self,
        d_model,
        num_attention_heads,
        max_seq_len,
    ):
        super(Layer, self).__init__()

        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert self.d_model % self.num_attention_heads == 0
        self.attention_head_size = int(d_model / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.projection = nn.Linear(d_model, d_model*3)

        self.attn_out = nn.Linear(d_model, d_model)
        self.norm1 = RMSNorm(d_model)

        self.positional_ff = MLP(d_model, d_model)
        self.norm2 = RMSNorm(d_model)

        self.rope = RotaryPositionEncoding(self.attention_head_size, max_seq_len)

    def split_heads(self, tensor, num_heads, attention_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attention_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)

    def merge_heads(self, tensor, num_heads, attention_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attention_head_size,)
        return tensor.view(new_shape)

    def attn(self, q, k, v, attention_mask):
        if attention_mask is not None:
            attention_mask = (attention_mask == 1).unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(-1, -1, q.size(2), -1)

        return nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0
        )

    def forward(self, x, attention_mask):
        residual = x

        q, k, v = self.projection(x).chunk(3, dim=-1)

        q = self.split_heads(q, self.num_attention_heads, self.attention_head_size)
        k = self.split_heads(k, self.num_attention_heads, self.attention_head_size)
        v = self.split_heads(v, self.num_attention_heads, self.attention_head_size)

        cos, sin = self.rope(q, seq_len=x.shape[1])
        q = self.rope.apply_rotary_pos_emb(q, cos, sin)
        k = self.rope.apply_rotary_pos_emb(k, cos, sin)

        attended_outputs = self.attn(q, k, v, attention_mask)
        attended_outputs = self.merge_heads(
            attended_outputs, self.num_attention_heads, self.attention_head_size
        )

        attended_outputs = self.attn_out(attended_outputs)
        x = self.norm1(attended_outputs + residual)

        residual = x
        x = self.positional_ff(x)
        x = self.norm2(x + residual)

        return x


class PETE(nn.Module):
    """
    Main PETE class that combines all components.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_hidden_layers,
        num_attention_heads,
        max_seq_len,
        permute_tokens: bool = False,
        random_embeddings: bool = False,
        index_mode: str = "raw",
        index_scale: float = 1.0,
        rff_sigma: Optional[float] = None,
    ):
        super(PETE, self).__init__()
        self.expansion = PolynomialBlock(
            max_seq_len,
            d_model,
            vocab_size=vocab_size,
            permute_tokens=permute_tokens,
            random_embeddings=random_embeddings,
            index_mode=index_mode,
            index_scale=index_scale,
            rff_sigma=rff_sigma,
        )
        self.mlp = nn.Linear(d_model, d_model)
        self.norm = RMSNorm(d_model)
        self.d_model = d_model
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.layers = nn.ModuleList(
            [
                Layer(
                    d_model,
                    num_attention_heads,
                    max_seq_len,
                )
                for _ in range(num_hidden_layers)
            ]
        )

        self.pooler = nn.Sequential(
            OrderedDict(
                [
                    ("dense", nn.Linear(d_model, d_model)),
                    ("activation", nn.Tanh()),
                ]
            )
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        x_polynomials = self.expansion(input_ids)
        x = self.norm(self.mlp(x_polynomials) + x_polynomials)

        for Layer in self.layers:
            x = Layer(x, attention_mask)

        return (x, self.pooler(x.mean(axis=1)))
