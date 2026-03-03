import argparse
import csv
import statistics
import time
from pathlib import Path

import torch

import attention_ext
from baseline_attention import attention_baseline


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

    naive_err = (naive - reference).abs().max().item()
    tiled_err = (tiled - reference).abs().max().item()
    naive_ok = torch.allclose(naive, reference, rtol=1e-3, atol=1e-3)
    tiled_ok = torch.allclose(tiled, reference, rtol=1e-3, atol=1e-3)
    return {
        "L": L,
        "d": d,
        "naive_max_abs_err": naive_err,
        "tiled_max_abs_err": tiled_err,
        "naive_ok": naive_ok,
        "tiled_ok": tiled_ok,
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
            f"tiled err={result['tiled_max_abs_err']:.6f}"
        )
        if not (result["naive_ok"] and result["tiled_ok"]):
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

    return [
        {"L": L, "d": d, "method": "cpu_baseline", "mean_ms": cpu_mean, "std_ms": cpu_std},
        {"L": L, "d": d, "method": "gpu_naive", "mean_ms": naive_mean, "std_ms": naive_std},
        {"L": L, "d": d, "method": "gpu_tiled", "mean_ms": tiled_mean, "std_ms": tiled_std},
    ]


def write_csv(rows, device_name, cuda_version, torch_version):
    RESULTS_DIR.mkdir(exist_ok=True)
    fieldnames = ["L", "d", "method", "mean_ms", "std_ms", "device_name", "cuda_version", "torch_version"]
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
    for row in rows:
        if row["method"] == "cpu_baseline":
            continue
        speedup = cpu_lookup[(row["L"], row["d"])] / row["mean_ms"]
        if speedup > best_speedup:
            best_speedup = speedup
            best_row = {**row, "speedup": speedup}

    lines = [
        "# Benchmark Summary",
        "",
        f"- GPU model name: {device_name}",
        f"- CUDA version: {cuda_version}",
        f"- Best speedup observed: {best_row['speedup']:.2f}x using {best_row['method']} at L={best_row['L']}, d={best_row['d']}" if best_row else "- Best speedup observed: not available",
        "- gpu_tiled optimizations: shared-memory tiling for QK^T and P@V, plus warp-shuffle row reductions for softmax max/sum.",
        f"- Max L tested: {max(row['L'] for row in rows) if rows else 'N/A'}",
        "- Memory notes: both GPU kernels materialize the LxL score/probability matrices in global memory.",
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
