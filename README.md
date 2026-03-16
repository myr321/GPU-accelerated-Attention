# CUDA-Accelerated Transformer Attention Forward Pass

This project implements a forward-only scaled dot-product attention kernel for batch size 1, single-head, float32 tensors of shape `[L, d]`.

The baseline function is copied from the Harvard NLP Annotated Transformer reference, with only the minimal edits required to remove mask/dropout and operate on `[L, d]` tensors.

```text
https://nlp.seas.harvard.edu/2018/04/03/attention.html
https://gist.github.com/Kaixhin/dc6f73099334a5d41d20804e70ae7f7b
```

## Inputs

Core attention inputs:

- `Q`, `K`, `V` tensors with shape `[L, d]`
- all three must have the same shape
- custom CUDA kernels expect CUDA tensors, `float32`, and contiguous layout
- batch size and number of heads are both fixed to `1`,
- no mask input
- no dropout input

Example:

```python
L, d = 256, 128
q = torch.randn(L, d, device="cuda", dtype=torch.float32).contiguous()
k = torch.randn(L, d, device="cuda", dtype=torch.float32).contiguous()
v = torch.randn(L, d, device="cuda", dtype=torch.float32).contiguous()
```

Benchmark script inputs:

- `--Ls`: sequence lengths to test
- `--ds`: hidden sizes to test
- `--warmup`: warmup iterations per shape
- `--iters`: timed iterations per shape

(Randomly generated matrices)

## Implemented Versions

- `cpu_baseline`: the copied PyTorch baseline running on CPU tensors.
- `gpu_naive`: three CUDA kernels using global-memory matmuls plus a shared-memory row softmax reduction.
- `gpu_tiled`: shared-memory tiled matmuls for `QK^T` and `P@V`, plus warp-shuffle reductions for the row-wise softmax max and sum.
- `gpu_fused_softmax_pv`: tiled `QK^T`, followed by a fused softmax + `P@V` kernel that avoids materializing the probability matrix.
- `gpu_official_pytorch`: an official reference implemented in `official_attention.py`, which wraps PyTorch GPU attention paths and picks the faster path for comparison.

## Constraints

- batch size = 1
- single attention head
- float32 only
- contiguous tensors only
- no mask, no dropout

## Setup (in Lab Machine)
CUDA and gcc environment:

```bash
source ~hardav/cuda/cuda-env.sh
```
```bash
source /opt/rh/gcc-toolset-13/enable
```

Create a Python environment:

```bash
cd .../Project
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

The build script defaults `TORCH_CUDA_ARCH_LIST` to `7.5` for the RTX 2060 GPU.
It also auto-selects a newer `gcc-toolset` if the default `g++` is older than PyTorch requires.

## Run Correctness Checks

```bash
python bench.py --check
```

This compares all custom CUDA implementations against the PyTorch baseline with:

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

Benchmark outputs in:

- `results/bench.csv`

## Generate Plots

```bash
python plot.py
```

Outputs in:

- `results/runtime_vs_L.png`
- `results/speedup_vs_L.png`
