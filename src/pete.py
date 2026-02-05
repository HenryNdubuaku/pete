import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

torch.manual_seed(0)
np.random.seed(0)


class PolynomialBlock(nn.Module):
    """
    Pure PyTorch implementation of Fourier embeddings.

    Transforms token IDs into dense embeddings using sinusoidal functions
    at different frequencies, similar to positional encodings but applied
    to token values.

    Supports ablations:
    - permute_tokens: Randomly permute token IDs before embedding
    - random_embeddings: Use fixed Gaussian random projection instead of Fourier
    """

    def __init__(
        self,
        max_seq_len: int,
        d_model: int,
        base: float = 10000.0,
        vocab_size: int = None,
        permute_tokens: bool = False,
        random_embeddings: bool = False,
        seed: int = 42,
    ):
        super(PolynomialBlock, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.base = base
        self.permute_tokens = permute_tokens
        self.random_embeddings = random_embeddings

        if random_embeddings:
            # Fixed Gaussian random projection: x -> x * W
            generator = torch.Generator().manual_seed(seed)
            random_proj = torch.randn(1, d_model, generator=generator)
            self.register_buffer("random_proj", random_proj)
        else:
            # Fourier frequencies
            inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
            self.register_buffer("inv_freq", inv_freq)

        if permute_tokens:
            assert vocab_size is not None, "vocab_size required when permute_tokens=True"
            generator = torch.Generator().manual_seed(seed)
            permutation = torch.randperm(vocab_size, generator=generator)
            self.register_buffer("permutation", permutation)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)

        Returns:
            Embeddings of shape (batch_size, seq_len, d_model)
        """
        if self.permute_tokens:
            input_ids = self.permutation[input_ids]

        input_ids = input_ids.float()

        # Shape: (batch_size, seq_len, 1)
        x = input_ids.unsqueeze(-1)

        if self.random_embeddings:
            # Random Gaussian projection: x * W
            embeddings = x * self.random_proj.to(x.device)
        else:
            # Fourier features
            freqs = x * self.inv_freq.to(x.device)
            sin_emb = torch.sin(freqs)
            cos_emb = torch.cos(freqs)
            embeddings = torch.cat([sin_emb, cos_emb], dim=-1)

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
        dot_product = torch.matmul(q, k.transpose(-1, -2))
        scaled_dot_product = dot_product / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_mask = attention_mask == 1
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scaled_dot_product = torch.where(
                attention_mask,
                scaled_dot_product,
                torch.tensor(float("-inf"), device=q.device),
            )

        attention_weights = nn.functional.softmax(scaled_dot_product, dim=-1)
        return torch.matmul(attention_weights, v)

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
        x = self.norm2(attended_outputs + residual)

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
    ):
        super(PETE, self).__init__()
        self.expansion = PolynomialBlock(
            max_seq_len,
            d_model,
            vocab_size=vocab_size,
            permute_tokens=permute_tokens,
            random_embeddings=random_embeddings,
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
