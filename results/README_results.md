# Benchmark Summary

- GPU model name: NVIDIA GeForce RTX 2060 SUPER
- CUDA version: 12.8
- Best speedup observed: 14.19x using gpu_official_pytorch at L=1024, d=64
- gpu_tiled optimizations: shared-memory tiling for QK^T and P@V, plus warp-shuffle row reductions for softmax max/sum.
- gpu_fused_softmax_pv optimizations: shared-memory tiled QK^T plus a fused softmax + P@V kernel that avoids materializing the probability matrix.
- gpu_official_pytorch implementation: wraps official PyTorch GPU attention paths and uses an empirical shape heuristic to choose between default scaled_dot_product_attention and eager GEMM + softmax.
- gpu_official_pytorch machine note: this sm75 RTX 2060 SUPER cannot use the newest fused SDPA backend available on newer GPUs, so the official comparison uses the fastest supported path on this lab machine.
- gpu_official_pytorch variant counts: {'sdpa_default': 11, 'eager_gemm_softmax': 9}
- Best gpu_official_pytorch speedup: 14.19x at L=1024, d=64 using eager_gemm_softmax
- Max L tested: 1024
- Memory notes: gpu_naive and gpu_tiled materialize both LxL score/probability matrices; gpu_fused_softmax_pv keeps scores but avoids writing the probability matrix; gpu_official_pytorch uses PyTorch library kernels instead of custom global buffers in this project code.
- Limitations: batch=1, single head, float32 only, contiguous [L, d] tensors only.
