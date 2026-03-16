# Official / industrial attention reference used in the benchmark.
#
# References:
# - https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
# - https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html
# - https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
#
# This file does not vendor external code. It wraps the official PyTorch
# attention implementations that are available on this machine and selects the
# faster path for the project workload on the RTX 2060 SUPER lab GPU.

import torch
import torch.nn.functional as F

from baseline_attention import attention_baseline


def _check_inputs(q, k, v):
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("Official GPU attention expects CUDA tensors.")
    if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        raise ValueError("Official GPU attention expects float32 tensors.")
    if q.dim() != 2 or k.dim() != 2 or v.dim() != 2:
        raise ValueError("Official GPU attention expects [L, d] tensors.")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q, K, and V must have the same shape.")


def attention_official_sdpa(q, k, v):
    _check_inputs(q, k, v)
    out = F.scaled_dot_product_attention(
        q.unsqueeze(0).unsqueeze(0),
        k.unsqueeze(0).unsqueeze(0),
        v.unsqueeze(0).unsqueeze(0),
        dropout_p=0.0,
        is_causal=False,
    )
    return out.squeeze(0).squeeze(0).contiguous()


def attention_official_eager(q, k, v):
    _check_inputs(q, k, v)
    return attention_baseline(q, k, v)


def select_official_variant(L, d):
    # Empirical heuristic for the RTX 2060 SUPER lab machine:
    # default SDPA wins on the smaller shapes in this project, while the eager
    # GEMM + softmax path wins once the problem gets larger on this sm75 GPU.
    if L <= 256 and L * d <= 32768:
        return "sdpa_default"
    return "eager_gemm_softmax"


def attention_official_best(q, k, v):
    variant = select_official_variant(q.size(0), q.size(1))
    if variant == "sdpa_default":
        return attention_official_sdpa(q, k, v)
    return attention_official_eager(q, k, v)
