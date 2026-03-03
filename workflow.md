# Project Workflow and Implementation Notes

This file is a handoff note for reading the repository quickly, understanding the implementation logic, and turning the project into slides or a report.

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

The project compares three methods:

- `cpu_baseline`: PyTorch baseline on CPU
- `gpu_naive`: custom CUDA implementation with simple kernels
- `gpu_tiled`: custom CUDA implementation with explicit GPU optimizations

## 2. Main Files

- `baseline_attention.py`
  - PyTorch baseline copied from the Harvard NLP Annotated Transformer reference, with only minimal edits for this project format.
- `attention_ext/attention.cpp`
  - C++ / PyBind binding layer.
  - Validates inputs and exposes two Python-callable CUDA functions.
- `attention_ext/attention_cuda.cu`
  - All CUDA kernels and launch code.
- `bench.py`
  - Correctness checking and benchmarking.
- `plot.py`
  - Reads CSV results and generates runtime / speedup plots.
- `results/`
  - Benchmark CSV, plots, and summary.

## 3. End-to-End Computation Flow

The attention computation is always split into three stages:

1. Compute score matrix `S = QK^T / sqrt(d)`
2. Compute row-wise probability matrix `P = softmax(S)`
3. Compute output `O = PV`

The repository keeps this three-stage structure in both CUDA versions. This is intentional:

- it matches the mathematical formula directly
- it is easy to debug
- it is easy to compare `naive` and `tiled`
- it is easy to explain in a presentation

The cost is that both GPU versions materialize:

- `scores` with shape `[L, L]`
- `probs` with shape `[L, L]`

in global memory, instead of fusing everything into one kernel.

## 4. Baseline Logic

File: `baseline_attention.py`

The baseline function is:

```python
def attention_baseline(Q, K, V):
    d = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
    P = F.softmax(scores, dim=-1)
    return torch.matmul(P, V)
```

This baseline serves two roles:

- CPU performance baseline
- GPU correctness reference when run on CUDA tensors

The project does not invent a new baseline; it follows the external reference required by the assignment.

## 5. Python to CUDA Binding Flow

File: `attention_ext/attention.cpp`

This layer does three things:

1. Validate that `Q`, `K`, and `V` are:
   - CUDA tensors
   - float32
   - contiguous
   - shape `[L, d]`
   - on the same device

2. Use `c10::cuda::CUDAGuard` so the kernels run on the correct GPU.

3. Expose two callable functions to Python:
   - `attention_forward_naive`
   - `attention_forward_tiled`

This binding file does not contain the math itself. It is only the safety and dispatch layer between Python and CUDA.

## 6. CUDA Version A: gpu_naive

File: `attention_ext/attention_cuda.cu`

### 6.1 `qk_naive_kernel`

Purpose:

- compute `scores = QK^T / sqrt(d)`

Mapping:

- one thread computes one output element `scores[row, col]`

Implementation idea:

- loop over `d`
- accumulate dot product
- apply scale `1 / sqrt(d)`

This version uses global memory directly. It is simple and readable, but it re-loads the same data many times.

### 6.2 `softmax_naive_kernel`

Purpose:

- compute row-wise stable softmax

Mapping:

- one block handles one row

Implementation idea:

1. reduce row max using shared memory
2. compute `exp(score - row_max)`
3. reduce row sum using shared memory
4. divide each entry by the row sum

Even though this is the "naive" version, it still uses shared-memory reduction for correctness and simplicity.

### 6.3 `pv_naive_kernel`

Purpose:

- compute `out = P @ V`

Mapping:

- one thread computes one output element `out[row, col]`

Implementation idea:

- loop over `L`
- accumulate `P[row, k] * V[k, col]`

## 7. CUDA Version B: gpu_tiled

The optimized version keeps the same three-stage algorithm, but changes how the expensive parts are executed.

### 7.1 Shared-memory tiled matmul for `QK^T`

Kernel:

- `qk_tiled_kernel`

Optimization method:

- split the matrix multiply into `16 x 16` tiles
- load one tile of `Q` and one tile of `K` into shared memory
- synchronize threads
- reuse the shared tile data for multiple multiply-add operations

Benefits:

- much less global memory traffic
- better data reuse
- more regular memory access

This is the main optimization over `qk_naive_kernel`.

### 7.2 Shared-memory tiled matmul for `P @ V`

Kernel:

- `pv_tiled_kernel`

Optimization method:

- same tiling idea as above
- use shared memory to stage blocks from `P` and `V`
- accumulate partial products from tiles instead of reading every value from global memory repeatedly

Benefits:

- lower memory bandwidth pressure
- improved reuse of `V` and `P` tile data

### 7.3 Warp-level reduction for softmax

Kernel:

- `softmax_warp_kernel`

Helper device functions:

- `warp_reduce_max`
- `warp_reduce_sum`

Optimization method:

- each thread scans part of one row
- each warp performs a local reduction using `__shfl_down_sync`
- each warp writes one partial result into shared memory
- the first warp performs a second-stage reduction over warp partials

This is a two-stage reduction:

1. reduce inside each warp
2. reduce across warps

Benefits compared with the naive softmax reduction:

- fewer shared-memory accesses
- fewer synchronization costs
- better use of warp execution behavior on the GPU

## 8. GPU Optimization Summary

The actual GPU optimization methods used in this project are:

1. Thread-level parallelism for matrix output elements
   - one thread per output element in naive matmul kernels

2. Shared-memory tiling
   - used in `qk_tiled_kernel`
   - used in `pv_tiled_kernel`

3. Warp-level reduction with shuffle instructions
   - used in `softmax_warp_kernel`
   - implemented with `__shfl_down_sync`

4. Two-stage reduction
   - first inside each warp
   - then across warps using shared memory

5. Stable softmax
   - subtract row max before exponentiation
   - prevents overflow and keeps results aligned with PyTorch

6. Small kernel-side simplifications
   - precompute `1 / sqrt(d)` once before launching kernels
   - assume contiguous float32 tensors so indexing is simple

7. GPU-specific build targeting
   - compile for compute capability `7.5`
   - this matches the RTX 2060 / RTX 2060 SUPER class hardware used for the project

## 9. What Was Intentionally Not Implemented

To keep the project small, clear, and robust, the following were intentionally not included:

- fused softmax + `P @ V`
- FlashAttention-style online softmax
- Tensor Cores
- half precision / FP16 / BF16
- backward pass
- multi-head attention
- batched input

This keeps the code easier to explain and easier to validate.

## 10. Kernel Launch Workflow

Both CUDA entry points follow the same host-side structure:

1. Read `L` and `d` from input tensor shape
2. Allocate:
   - `scores [L, L]`
   - `probs [L, L]`
   - `out [L, d]`
3. Launch:
   - score kernel
   - softmax kernel
   - output kernel
4. Check CUDA launch status after each kernel
5. Return `out`

This structure appears in:

- `attention_forward_naive_cuda`
- `attention_forward_tiled_cuda`

The only difference is which kernels they call.

## 11. Correctness Workflow

Correctness is checked in `bench.py`.

Method:

1. create random `Q`, `K`, `V` with fixed seed
2. compute PyTorch GPU reference with `attention_baseline`
3. compute custom outputs with:
   - `attention_forward_naive`
   - `attention_forward_tiled`
4. compare with:

```python
torch.allclose(..., rtol=1e-3, atol=1e-3)
```

The script also prints maximum absolute error.

Observed result:

- both CUDA versions passed the correctness checks

## 12. Benchmark Workflow

Benchmarking is also handled by `bench.py`.

Process:

1. detect GPU name
2. warm up each method
3. time GPU methods with CUDA events
4. time CPU baseline with `time.perf_counter()`
5. save rows into `results/bench.csv`
6. summarize best speedup in `results/README_results.md`

Test grid used in the benchmark:

- `L in [64, 128, 256, 512, 1024]`
- `d in [32, 64, 128, 256]`

Plotting is handled by `plot.py`, which generates:

- `results/runtime_vs_L.png`
- `results/speedup_vs_L.png`

## 13. Final Measured Result

From `results/README_results.md`:

- GPU: `NVIDIA GeForce RTX 2060 SUPER`
- CUDA version: `12.8`
- Best speedup observed: `2461.38x`
- Best case: `gpu_tiled` at `L = 128`, `d = 128`

Important note for presentation:

- the large speedup comes from comparing against a CPU baseline built from high-level PyTorch ops
- the key story is not only "the number is large"
- the key story is that the optimized CUDA version is consistently faster than the CPU baseline and faster than the naive GPU version for larger workloads

## 14. Suggested PPT Storyline

If this repository is given to another model or used to make slides, the cleanest presentation order is:

1. Problem statement
   - scaled dot-product attention is expensive because it contains two matrix multiplies plus softmax

2. Scope simplification
   - batch = 1, single head, float32, forward only

3. Baseline
   - PyTorch reference copied from Harvard Annotated Transformer

4. CUDA version A: naive
   - one thread per output element
   - shared-memory row softmax

5. CUDA version B: tiled
   - shared-memory tiled matmuls
   - warp-level reduction for softmax

6. Correctness
   - matched PyTorch GPU reference within `rtol=1e-3`, `atol=1e-3`

7. Performance
   - runtime plot
   - speedup plot
   - best observed speedup

8. Limitations and future work
   - no kernel fusion
   - no backward pass
   - no tensor-core usage

## 15. Short Takeaway

The implementation goal of this project is not "maximum possible attention speed". The goal is:

- build a correct custom CUDA attention forward pass
- provide a clear baseline
- show two GPU implementation levels
- demonstrate concrete GPU optimization ideas
- produce reproducible benchmark results and plots

That goal is satisfied by the current repository.
