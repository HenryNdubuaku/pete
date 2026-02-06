"""
Triton kernels for polynomial embeddings.
Replaces the C++/CUDA kernels with a single Python file.

Supports: Fourier, Chebyshev, Legendre, Hermite, Laguerre
"""

import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fourier_kernel(
    input_ptr,
    output_ptr,
    max_seq_len,
    d_model,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    token_idx = offsets // d_model
    freq_idx = offsets % d_model

    x_raw = tl.load(input_ptr + token_idx, mask=mask)
    x = 2.0 * (x_raw / (max_seq_len - 1)) - 1.0

    freq = (freq_idx // 2) + 1
    angle = freq.to(tl.float32) * x * 3.141592653589793

    is_even = (freq_idx % 2) == 0
    result = tl.where(is_even, tl.sin(angle), tl.cos(angle))

    tl.store(output_ptr + offsets, result, mask=mask)


def fourier(input_ids: torch.Tensor, max_seq_len: int, d_model: int) -> list:
    """Fourier polynomial embedding."""
    assert input_ids.is_cuda, "Input must be on CUDA"
    batch_size, seq_len = input_ids.shape
    total_elements = batch_size * seq_len * d_model

    output = torch.zeros(batch_size, seq_len, d_model, device=input_ids.device, dtype=input_ids.dtype)

    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _fourier_kernel[grid](
        input_ids, output, max_seq_len, d_model, total_elements, BLOCK_SIZE
    )
    return [output]


@triton.jit
def _chebyshev_kernel(
    input_ptr,
    output_ptr,
    max_seq_len,
    d_model,
    seq_stride,
    total_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_offsets < total_tokens

    x_raw = tl.load(input_ptr + token_offsets, mask=mask)
    x = 2.0 * (x_raw / (max_seq_len - 1)) - 1.0

    base_out = token_offsets * d_model

    # T_0 = 1
    tl.store(output_ptr + base_out, tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32), mask=mask)

    # T_1 = x (if d_model > 1)
    if d_model > 1:
        tl.store(output_ptr + base_out + 1, x, mask=mask)

    # Recurrence: T_n = 2*x*T_{n-1} - T_{n-2}
    T_prev2 = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    T_prev1 = x
    for n in range(2, d_model):
        T_n = 2.0 * x * T_prev1 - T_prev2
        tl.store(output_ptr + base_out + n, T_n, mask=mask)
        T_prev2 = T_prev1
        T_prev1 = T_n


def chebyshev(input_ids: torch.Tensor, max_seq_len: int, d_model: int) -> list:
    """Chebyshev polynomial embedding."""
    assert input_ids.is_cuda, "Input must be on CUDA"
    batch_size, seq_len = input_ids.shape
    total_tokens = batch_size * seq_len

    output = torch.zeros(batch_size, seq_len, d_model, device=input_ids.device, dtype=input_ids.dtype)

    BLOCK_SIZE = 256
    grid = ((total_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _chebyshev_kernel[grid](
        input_ids, output, max_seq_len, d_model, d_model, total_tokens, BLOCK_SIZE
    )
    return [output]


@triton.jit
def _legendre_kernel(
    input_ptr,
    output_ptr,
    max_seq_len,
    d_model,
    total_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_offsets < total_tokens

    x_raw = tl.load(input_ptr + token_offsets, mask=mask)
    x = 2.0 * (x_raw / (max_seq_len - 1)) - 1.0

    base_out = token_offsets * d_model

    # P_0 = 1
    tl.store(output_ptr + base_out, tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32), mask=mask)

    if d_model > 1:
        # P_1 = x
        tl.store(output_ptr + base_out + 1, x, mask=mask)

    # Recurrence: P_{n+1} = ((2n+1)*x*P_n - n*P_{n-1}) / (n+1)
    P_prev2 = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    P_prev1 = x
    for n in range(1, d_model - 1):
        n_f = float(n)
        P_n = ((2.0 * n_f + 1.0) * x * P_prev1 - n_f * P_prev2) / (n_f + 1.0)
        tl.store(output_ptr + base_out + n + 1, P_n, mask=mask)
        P_prev2 = P_prev1
        P_prev1 = P_n


def legendre(input_ids: torch.Tensor, max_seq_len: int, d_model: int) -> list:
    """Legendre polynomial embedding."""
    assert input_ids.is_cuda, "Input must be on CUDA"
    batch_size, seq_len = input_ids.shape
    total_tokens = batch_size * seq_len

    output = torch.zeros(batch_size, seq_len, d_model, device=input_ids.device, dtype=input_ids.dtype)

    BLOCK_SIZE = 256
    grid = ((total_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _legendre_kernel[grid](
        input_ids, output, max_seq_len, d_model, total_tokens, BLOCK_SIZE
    )
    return [output]


@triton.jit
def _hermite_kernel(
    input_ptr,
    output_ptr,
    max_seq_len,
    d_model,
    total_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_offsets < total_tokens

    x_raw = tl.load(input_ptr + token_offsets, mask=mask)
    x = 2.0 * (x_raw / (max_seq_len - 1)) - 1.0

    base_out = token_offsets * d_model

    # H_0 = 1
    tl.store(output_ptr + base_out, tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32), mask=mask)

    if d_model > 1:
        # H_1 = 2x
        tl.store(output_ptr + base_out + 1, 2.0 * x, mask=mask)

    # Recurrence: H_n = 2*x*H_{n-1} - 2*(n-1)*H_{n-2}
    H_prev2 = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    H_prev1 = 2.0 * x
    for n in range(2, d_model):
        n_f = float(n)
        H_n = 2.0 * x * H_prev1 - 2.0 * (n_f - 1.0) * H_prev2
        tl.store(output_ptr + base_out + n, H_n, mask=mask)
        H_prev2 = H_prev1
        H_prev1 = H_n


def hermite(input_ids: torch.Tensor, max_seq_len: int, d_model: int) -> list:
    """Hermite polynomial embedding."""
    assert input_ids.is_cuda, "Input must be on CUDA"
    batch_size, seq_len = input_ids.shape
    total_tokens = batch_size * seq_len

    output = torch.zeros(batch_size, seq_len, d_model, device=input_ids.device, dtype=input_ids.dtype)

    BLOCK_SIZE = 256
    grid = ((total_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _hermite_kernel[grid](
        input_ids, output, max_seq_len, d_model, total_tokens, BLOCK_SIZE
    )
    return [output]


@triton.jit
def _laguerre_kernel(
    input_ptr,
    output_ptr,
    max_seq_len,
    d_model,
    total_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_offsets < total_tokens

    x_raw = tl.load(input_ptr + token_offsets, mask=mask)
    # Laguerre uses x in [0, inf), so we normalize to [0, 2] range
    x = 2.0 * (x_raw / (max_seq_len - 1))

    base_out = token_offsets * d_model

    # L_0 = 1
    tl.store(output_ptr + base_out, tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32), mask=mask)

    if d_model > 1:
        # L_1 = 1 - x
        tl.store(output_ptr + base_out + 1, 1.0 - x, mask=mask)

    # Recurrence: L_n = ((2n-1-x)*L_{n-1} - (n-1)*L_{n-2}) / n
    L_prev2 = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    L_prev1 = 1.0 - x
    for n in range(2, d_model):
        n_f = float(n)
        L_n = ((2.0 * n_f - 1.0 - x) * L_prev1 - (n_f - 1.0) * L_prev2) / n_f
        tl.store(output_ptr + base_out + n, L_n, mask=mask)
        L_prev2 = L_prev1
        L_prev1 = L_n


def laguerre(input_ids: torch.Tensor, max_seq_len: int, d_model: int) -> list:
    """Laguerre polynomial embedding."""
    assert input_ids.is_cuda, "Input must be on CUDA"
    batch_size, seq_len = input_ids.shape
    total_tokens = batch_size * seq_len

    output = torch.zeros(batch_size, seq_len, d_model, device=input_ids.device, dtype=input_ids.dtype)

    BLOCK_SIZE = 256
    grid = ((total_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _laguerre_kernel[grid](
        input_ids, output, max_seq_len, d_model, total_tokens, BLOCK_SIZE
    )
    return [output]
