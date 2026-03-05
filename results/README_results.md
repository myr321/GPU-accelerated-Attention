# Benchmark Summary

- GPU model name: NVIDIA GeForce RTX 2060 SUPER
- CUDA version: 12.8
- Best speedup observed: 4.49x using gpu_tiled at L=1024, d=64
- gpu_tiled optimizations: shared-memory tiling for QK^T and P@V, plus warp-shuffle row reductions for softmax max/sum.
- Max L tested: 1024
- Memory notes: both GPU kernels materialize the LxL score/probability matrices in global memory.
- Limitations: batch=1, single head, float32 only, contiguous [L, d] tensors only.
