import argparse
import csv
import statistics
import time
from pathlib import Path

import torch

import attention_ext
from baseline_attention import attention_baseline
from official_attention import attention_official_best, select_official_variant


DEFAULT_LS = [64, 128, 256, 512, 1024]
DEFAULT_DS = [32, 64, 128, 256]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "bench.csv"
SUMMARY_PATH = RESULTS_DIR / "README_results.md"


def parse_int_list(values):
    if not values:
        return []
    parsed = []
    for value in values:
        parsed.append(int(value))
    return parsed


def measure_cpu(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.mean(samples), statistics.pstdev(samples) if len(samples) > 1 else 0.0


def measure_cuda(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.mean(samples), statistics.pstdev(samples) if len(samples) > 1 else 0.0


def run_correctness_case(L, d):
    torch.manual_seed(0)
    q = torch.randn(L, d, device="cuda", dtype=torch.float32).contiguous()
    k = torch.randn(L, d, device="cuda", dtype=torch.float32).contiguous()
    v = torch.randn(L, d, device="cuda", dtype=torch.float32).contiguous()

    with torch.no_grad():
        reference = attention_baseline(q, k, v)
        naive = attention_ext.attention_forward_naive(q, k, v)
        tiled = attention_ext.attention_forward_tiled(q, k, v)
        fused = attention_ext.attention_forward_fused_softmax_pv(q, k, v)
        official = attention_official_best(q, k, v)

    naive_err = (naive - reference).abs().max().item()
    tiled_err = (tiled - reference).abs().max().item()
    fused_err = (fused - reference).abs().max().item()
    official_err = (official - reference).abs().max().item()
    naive_ok = torch.allclose(naive, reference, rtol=1e-3, atol=1e-3)
    tiled_ok = torch.allclose(tiled, reference, rtol=1e-3, atol=1e-3)
    fused_ok = torch.allclose(fused, reference, rtol=1e-3, atol=1e-3)
    official_ok = torch.allclose(official, reference, rtol=1e-3, atol=1e-3)
    return {
        "L": L,
        "d": d,
        "naive_max_abs_err": naive_err,
        "tiled_max_abs_err": tiled_err,
        "fused_max_abs_err": fused_err,
        "official_max_abs_err": official_err,
        "naive_ok": naive_ok,
        "tiled_ok": tiled_ok,
        "fused_ok": fused_ok,
        "official_ok": official_ok,
    }


def run_correctness():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a CUDA-enabled PyTorch build first.")

    cases = [(64, 32), (127, 64), (256, 128)]
    print("Running correctness checks...")
    all_ok = True
    for L, d in cases:
        result = run_correctness_case(L, d)
        print(
            f"  L={L:4d}, d={d:3d} | "
            f"naive err={result['naive_max_abs_err']:.6f}, "
            f"tiled err={result['tiled_max_abs_err']:.6f}, "
            f"fused err={result['fused_max_abs_err']:.6f}, "
            f"official err={result['official_max_abs_err']:.6f}"
        )
        if not (result["naive_ok"] and result["tiled_ok"] and result["fused_ok"] and result["official_ok"]):
            all_ok = False
    if not all_ok:
        raise RuntimeError("Correctness checks failed.")
    print("Correctness checks passed.")


def benchmark_shape(L, d, warmup, iters):
    q_cpu = torch.randn(L, d, dtype=torch.float32).contiguous()
    k_cpu = torch.randn(L, d, dtype=torch.float32).contiguous()
    v_cpu = torch.randn(L, d, dtype=torch.float32).contiguous()
    q_gpu = q_cpu.cuda()
    k_gpu = k_cpu.cuda()
    v_gpu = v_cpu.cuda()

    with torch.no_grad():
        cpu_mean, cpu_std = measure_cpu(lambda: attention_baseline(q_cpu, k_cpu, v_cpu), warmup, iters)
        naive_mean, naive_std = measure_cuda(
            lambda: attention_ext.attention_forward_naive(q_gpu, k_gpu, v_gpu), warmup, iters
        )
        tiled_mean, tiled_std = measure_cuda(
            lambda: attention_ext.attention_forward_tiled(q_gpu, k_gpu, v_gpu), warmup, iters
        )
        fused_mean, fused_std = measure_cuda(
            lambda: attention_ext.attention_forward_fused_softmax_pv(q_gpu, k_gpu, v_gpu), warmup, iters
        )
        official_variant = select_official_variant(L, d)
        official_mean, official_std = measure_cuda(
            lambda: attention_official_best(q_gpu, k_gpu, v_gpu), warmup, iters
        )

    return [
        {"L": L, "d": d, "method": "cpu_baseline", "variant": "", "mean_ms": cpu_mean, "std_ms": cpu_std},
        {"L": L, "d": d, "method": "gpu_naive", "variant": "", "mean_ms": naive_mean, "std_ms": naive_std},
        {"L": L, "d": d, "method": "gpu_tiled", "variant": "", "mean_ms": tiled_mean, "std_ms": tiled_std},
        {
            "L": L,
            "d": d,
            "method": "gpu_fused_softmax_pv",
            "variant": "",
            "mean_ms": fused_mean,
            "std_ms": fused_std,
        },
        {
            "L": L,
            "d": d,
            "method": "gpu_official_pytorch",
            "variant": official_variant,
            "mean_ms": official_mean,
            "std_ms": official_std,
        },
    ]


def write_csv(rows, device_name, cuda_version, torch_version):
    RESULTS_DIR.mkdir(exist_ok=True)
    fieldnames = [
        "L",
        "d",
        "method",
        "variant",
        "mean_ms",
        "std_ms",
        "device_name",
        "cuda_version",
        "torch_version",
    ]
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "device_name": device_name,
                    "cuda_version": cuda_version,
                    "torch_version": torch_version,
                }
            )


def write_summary(rows, device_name, cuda_version):
    cpu_lookup = {}
    for row in rows:
        if row["method"] == "cpu_baseline":
            cpu_lookup[(row["L"], row["d"])] = row["mean_ms"]

    best_speedup = 0.0
    best_row = None
    best_official = None
    official_variant_counts = {}
    for row in rows:
        if row["method"] == "cpu_baseline":
            continue
        speedup = cpu_lookup[(row["L"], row["d"])] / row["mean_ms"]
        if speedup > best_speedup:
            best_speedup = speedup
            best_row = {**row, "speedup": speedup}
        if row["method"] == "gpu_official_pytorch":
            official_variant_counts[row["variant"]] = official_variant_counts.get(row["variant"], 0) + 1
            if best_official is None or speedup > best_official["speedup"]:
                best_official = {**row, "speedup": speedup}

    lines = [
        "# Benchmark Summary",
        "",
        f"- GPU model name: {device_name}",
        f"- CUDA version: {cuda_version}",
        f"- Best speedup observed: {best_row['speedup']:.2f}x using {best_row['method']} at L={best_row['L']}, d={best_row['d']}" if best_row else "- Best speedup observed: not available",
        "- gpu_tiled optimizations: shared-memory tiling for QK^T and P@V, plus warp-shuffle row reductions for softmax max/sum.",
        "- gpu_fused_softmax_pv optimizations: shared-memory tiled QK^T plus a fused softmax + P@V kernel that avoids materializing the probability matrix.",
        "- gpu_official_pytorch implementation: wraps official PyTorch GPU attention paths and uses an empirical shape heuristic to choose between default scaled_dot_product_attention and eager GEMM + softmax.",
        "- gpu_official_pytorch machine note: this sm75 RTX 2060 SUPER cannot use the newest fused SDPA backend available on newer GPUs, so the official comparison uses the fastest supported path on this lab machine.",
        f"- gpu_official_pytorch variant counts: {official_variant_counts}" if official_variant_counts else "- gpu_official_pytorch variant counts: not available",
        f"- Best gpu_official_pytorch speedup: {best_official['speedup']:.2f}x at L={best_official['L']}, d={best_official['d']} using {best_official['variant']}" if best_official else "- Best gpu_official_pytorch speedup: not available",
        f"- Max L tested: {max(row['L'] for row in rows) if rows else 'N/A'}",
        "- Memory notes: gpu_naive and gpu_tiled materialize both LxL score/probability matrices; gpu_fused_softmax_pv keeps scores but avoids writing the probability matrix; gpu_official_pytorch uses PyTorch library kernels instead of custom global buffers in this project code.",
        "- Limitations: batch=1, single head, float32 only, contiguous [L, d] tensors only.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA attention forward kernels.")
    parser.add_argument("--check", action="store_true", help="Run correctness checks and exit.")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations per shape.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per shape.")
    parser.add_argument("--Ls", nargs="*", default=DEFAULT_LS, help="Sequence lengths to benchmark.")
    parser.add_argument("--ds", nargs="*", default=DEFAULT_DS, help="Hidden sizes to benchmark.")
    args = parser.parse_args()

    if args.check:
        run_correctness()
        return

    run_correctness()

    device_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda or "unknown"
    torch_version = torch.__version__
    Ls = parse_int_list(args.Ls)
    ds = parse_int_list(args.ds)

    rows = []
    print(f"Benchmarking on {device_name} (CUDA {cuda_version}, PyTorch {torch_version})")
    for d in ds:
        for L in Ls:
            print(f"  L={L:4d}, d={d:3d}")
            try:
                rows.extend(benchmark_shape(L, d, args.warmup, args.iters))
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"    Skipping remaining benchmarks after OOM at L={L}, d={d}")
                    torch.cuda.empty_cache()
                    break
                raise

    write_csv(rows, device_name, cuda_version, torch_version)
    write_summary(rows, device_name, cuda_version)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
