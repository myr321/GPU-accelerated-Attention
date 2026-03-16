# CUDA-Accelerated Transformer Attention Forward Pass

This project implements a forward-only scaled dot-product attention kernel for batch size 1, single-head, float32 tensors of shape `[L, d]`.

The baseline function is copied from the Harvard NLP Annotated Transformer reference, with only the minimal edits required to remove mask/dropout and operate on `[L, d]` tensors.

```text
https://nlp.seas.harvard.edu/2018/04/03/attention.html
https://gist.github.com/Kaixhin/dc6f73099334a5d41d20804e70ae7f7b
```

## Implemented Versions

- `cpu_baseline`: the copied PyTorch baseline running on CPU tensors.
- `gpu_naive`: three CUDA kernels using global-memory matmuls plus a shared-memory row softmax reduction.
- `gpu_tiled`: shared-memory tiled matmuls for `QK^T` and `P@V`, plus warp-shuffle reductions for the row-wise softmax max and sum.
- `gpu_fused_softmax_pv`: tiled `QK^T`, followed by a fused softmax + `P@V` kernel that avoids materializing the probability matrix.
- `gpu_official_pytorch`: an official / industrial reference implemented in `official_attention.py`, which wraps PyTorch GPU attention paths and picks the faster path for this lab machine.

Note:

- `gpu_official_pytorch` is the main production reference used in the plots and tables.

## Constraints

- batch size = 1
- single attention head
- float32 only
- contiguous tensors only
- no mask, no dropout

## Lab Machine Setup

Use the lab CUDA environment script before building:

```bash
source ~hardav/cuda/cuda-env.sh
```

If your shell still points at an older system compiler, you can also enable the newer toolchain explicitly:

```bash
source /opt/rh/gcc-toolset-13/enable
```

Create a Python environment. The system has Python 3.11 available:

```bash
cd /home/mic6954/CE468/Project
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build first, then the remaining Python dependencies:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install -r requirements.txt
```

## Build

Either editable install:

```bash
pip install -e .
```

Or build the extension in place:

```bash
python setup.py build_ext --inplace
```

The build script defaults `TORCH_CUDA_ARCH_LIST` to `7.5` for the RTX 2060 class GPU used in lab.
It also auto-selects a newer `gcc-toolset` if the default `g++` is older than PyTorch requires.

## Run Correctness Checks

```bash
python bench.py --check
```

This compares all custom CUDA implementations against the PyTorch GPU reference with:

```python
torch.allclose(..., rtol=1e-3, atol=1e-3)
```

Covered CUDA variants:

- `attention_forward_naive`
- `attention_forward_tiled`
- `attention_forward_fused_softmax_pv`

Covered benchmarked GPU paths:

- `attention_forward_naive`
- `attention_forward_tiled`
- `attention_forward_fused_softmax_pv`
- `attention_official_best`

## Run Benchmarks

```bash
python bench.py
```

Useful overrides while iterating:

```bash
python bench.py --iters 10 --warmup 5 --Ls 64 128 256 --ds 32 64
```

Benchmark outputs:

- `results/bench.csv`
- `results/README_results.md`

Notes:

- GPU timings use CUDA events.
- CPU baseline timings use `time.perf_counter()` because CUDA events cannot time CPU-only execution.
- The benchmark includes `gpu_naive`, `gpu_tiled`, `gpu_fused_softmax_pv`, and `gpu_official_pytorch`.
- `gpu_official_pytorch` uses a small empirical heuristic for this RTX 2060 SUPER machine:
  - default PyTorch `scaled_dot_product_attention` for smaller shapes
  - eager `QK^T -> softmax -> PV` with PyTorch GPU kernels for larger shapes
- The benchmark CSV includes a `variant` column so you can see which official path was selected at each `(L, d)`.

## Generate Plots

```bash
python plot.py
```

This writes:

- `results/runtime_vs_L.png`
- `results/speedup_vs_L.png`

## Package Submission

```bash
bash make_tarball.sh
```

This creates `final_submission.tar.gz` with source files, build instructions, and the `results/` folder.
