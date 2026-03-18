# Project Workflow and Implementation Notes

This file is a handoff note for quickly understanding the repository and turning it into slides or a report.

## 1. Project Goal

The project implements a simplified Transformer scaled dot-product attention forward pass:

`Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V`

Scope constraints:

- batch size = 1
- single head
- float32 only
- contiguous tensors only
- input shape = `[L, d]`
- no mask
- no dropout

The main comparison now uses five methods:

- `cpu_baseline`: PyTorch baseline on CPU
- `gpu_naive`: custom CUDA implementation with simple kernels
- `gpu_tiled`: custom CUDA implementation with shared-memory tiling and warp reductions
- `gpu_fused_softmax_pv`: custom CUDA implementation with a fused softmax + output stage
- `gpu_official_pytorch`: official / industrial PyTorch GPU reference

## 2. Main Files

- `baseline_attention.py`
  - PyTorch baseline copied from the Harvard NLP Annotated Transformer reference, with only minimal edits.
- `attention_ext/attention.cpp`
  - C++ / PyBind binding layer for the custom CUDA kernels.
- `attention_ext/attention_cuda.cu`
  - custom CUDA kernels for `gpu_naive`, `gpu_tiled`, and `gpu_fused_softmax_pv`.
- `official_attention.py`
  - official / industrial comparison wrapper built on top of PyTorch GPU attention APIs.
- `bench.py`
  - correctness checks and benchmarking.
- `plot.py`
  - generates runtime and speedup plots from the CSV.
- `results/`
  - benchmark CSV, summary, and plots.

## 3. End-to-End Computation Flow

The mathematical attention computation has three stages:

1. compute `S = QK^T / sqrt(d)`
2. compute `P = softmax(S)` row-wise
3. compute `O = PV`

The custom kernels keep this structure on purpose:

- it matches the formula directly
- it is easier to debug and validate
- it creates a clean optimization story: `naive -> tiled -> fused`

Memory implications:

- `gpu_naive` and `gpu_tiled` materialize both `scores [L, L]` and `probs [L, L]`
- `gpu_fused_softmax_pv` materializes `scores [L, L]` but avoids writing `probs`
- `gpu_official_pytorch` delegates storage and fusion details to PyTorch library kernels

## 4. Baseline Logic

File: `baseline_attention.py`

The baseline is:

```python
def attention_baseline(Q, K, V):
    d = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
    P = F.softmax(scores, dim=-1)
    return torch.matmul(P, V)
```

It serves two roles:

- CPU performance baseline
- GPU correctness reference when run on CUDA tensors

## 5. Python to CUDA Binding Flow

File: `attention_ext/attention.cpp`

This layer:

1. validates that `Q`, `K`, and `V` are CUDA, float32, contiguous, same-shaped `[L, d]` tensors
2. uses `c10::cuda::CUDAGuard` to launch on the correct GPU
3. exposes the custom CUDA entry points to Python

The binding file is only a validation and dispatch layer. The math lives in the CUDA files.

## 6. Custom Version A: gpu_naive

File: `attention_ext/attention_cuda.cu`

Kernels:

- `qk_naive_kernel`
- `softmax_naive_kernel`
- `pv_naive_kernel`

Logic:

- compute one output element per thread in the two matmul stages
- use a shared-memory row reduction for stable softmax
- read directly from global memory in the matmul stages

This version is straightforward and easy to explain, but it has poor data reuse.

## 7. Custom Version B: gpu_tiled

File: `attention_ext/attention_cuda.cu`

Kernels:

- `qk_tiled_kernel`
- `softmax_warp_kernel`
- `pv_tiled_kernel`

Optimization methods:

- shared-memory tiling for `QK^T`
- shared-memory tiling for `P @ V`
- warp-shuffle reduction for row-wise softmax max and sum
- two-stage reduction: first within a warp, then across warps

This is the strongest pure custom CUDA version in the repo and is usually the fastest custom kernel.

## 8. Custom Version C: gpu_fused_softmax_pv

File: `attention_ext/attention_cuda.cu`

Kernels:

- `qk_tiled_kernel`
- `softmax_pv_fused_kernel`

Optimization methods:

- reuse the tiled score kernel from `gpu_tiled`
- fuse softmax and output accumulation
- avoid materializing the probability matrix `P`

Tradeoff:

- lower memory traffic than the separate softmax + `P @ V` path
- simpler than a fully blockwise online-softmax implementation
- not always faster than `gpu_tiled`, because the `V` accumulation is less aggressively tiled

## 9. Official / Industrial Version: gpu_official_pytorch

File: `official_attention.py`

This version is not a custom CUDA kernel. It is the production reference used in the comparison.

It wraps two official PyTorch GPU paths:

- default `torch.nn.functional.scaled_dot_product_attention`
- eager `torch.matmul -> softmax -> torch.matmul` on CUDA tensors

Why both are used:

- on this RTX 2060 SUPER (`sm75`), the newest fused SDPA backend used on newer GPUs is not available
- the default SDPA path wins on smaller problem sizes in this project
- the eager GEMM + softmax path wins on larger sizes

The wrapper uses a small empirical heuristic for this machine:

- use SDPA when `L <= 256` and `L * d <= 32768`
- otherwise use eager GEMM + softmax

This makes `gpu_official_pytorch` the fastest practical official comparison point on the lab hardware without introducing an external dependency.

## 10. GPU Optimization Summary

The actual optimization methods used in the custom kernels are:

1. shared-memory tiling
2. warp-level reduction with `__shfl_down_sync`
3. two-stage reduction across warp and block scopes
4. stable softmax by subtracting the row max
5. partial kernel fusion for `softmax + P @ V`
6. compile targeting for compute capability `7.5`

The official version adds a different kind of optimization story:

- instead of hand-writing another custom kernel, it reuses production PyTorch kernels that are already backed by industrial GPU libraries

## 11. Correctness Workflow

Correctness is checked in `bench.py`.

Method:

1. create random `Q`, `K`, `V`
2. compute the PyTorch GPU reference with `attention_baseline`
3. compute outputs with:
   - `attention_forward_naive`
   - `attention_forward_tiled`
   - `attention_forward_fused_softmax_pv`
   - `attention_official_best`
4. compare with `torch.allclose(..., rtol=1e-3, atol=1e-3)`

The script also prints maximum absolute error per method.

## 12. Benchmark Workflow

Benchmarking is also handled by `bench.py`.

What it measures:

- CPU baseline runtime with `time.perf_counter()`
- GPU runtime with CUDA events

Benchmarked methods in the default CSV and plots:

- `cpu_baseline`
- `gpu_naive`
- `gpu_tiled`
- `gpu_fused_softmax_pv`
- `gpu_official_pytorch`

The CSV also stores a `variant` column:

- blank for the custom methods
- `sdpa_default` or `eager_gemm_softmax` for `gpu_official_pytorch`

## 13. Plotting and Presentation Story

The most presentation-friendly narrative is:

1. start from the mathematical baseline
2. show `gpu_naive` as the direct CUDA translation
3. show `gpu_tiled` as the main memory-reuse optimization
4. show `gpu_fused_softmax_pv` as the fusion attempt
5. end with `gpu_official_pytorch` as the industrial reference point

Recommended framing:

- the three custom versions demonstrate optimization ideas clearly
- the official PyTorch version shows what a production-quality library path can do on the same hardware
- the comparison stays focused on kernels that are actually present in the repo and used in the final plots
