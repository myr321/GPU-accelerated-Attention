import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "bench.csv"
RUNTIME_PLOT = RESULTS_DIR / "runtime_vs_L.png"
SPEEDUP_PLOT = RESULTS_DIR / "speedup_vs_L.png"
METHOD_STYLES = {
    "cpu_baseline": {"label": "cpu_baseline", "color": "#1b4965"},
    "gpu_naive": {"label": "gpu_naive", "color": "#ca6702"},
    "gpu_tiled": {"label": "gpu_tiled", "color": "#2a9d8f"},
}


def make_grid(num_panels):
    cols = 2
    rows = math.ceil(num_panels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    return fig, axes.flatten()


def plot_runtime(df):
    ds = sorted(df["d"].unique())
    fig, axes = make_grid(len(ds))

    for axis, d in zip(axes, ds):
        subset = df[df["d"] == d]
        for method in ["cpu_baseline", "gpu_naive", "gpu_tiled"]:
            method_df = subset[subset["method"] == method].sort_values("L")
            if method_df.empty:
                continue
            style = METHOD_STYLES[method]
            axis.plot(method_df["L"], method_df["mean_ms"], marker="o", linewidth=2, **style)
        axis.set_title(f"d = {d}")
        axis.set_xlabel("Sequence length L")
        axis.set_ylabel("Mean runtime (ms)")
        axis.grid(alpha=0.3)
        axis.legend()

    for axis in axes[len(ds):]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(RUNTIME_PLOT, dpi=200)
    plt.close(fig)


def plot_speedup(df):
    cpu_df = (
        df[df["method"] == "cpu_baseline"][["L", "d", "mean_ms"]]
        .rename(columns={"mean_ms": "cpu_mean_ms"})
        .sort_values(["d", "L"])
    )
    gpu_df = df[df["method"] != "cpu_baseline"].merge(cpu_df, on=["L", "d"], how="inner")
    gpu_df["speedup"] = gpu_df["cpu_mean_ms"] / gpu_df["mean_ms"]

    ds = sorted(gpu_df["d"].unique())
    fig, axes = make_grid(len(ds))

    for axis, d in zip(axes, ds):
        subset = gpu_df[gpu_df["d"] == d]
        for method in ["gpu_naive", "gpu_tiled"]:
            method_df = subset[subset["method"] == method].sort_values("L")
            if method_df.empty:
                continue
            style = METHOD_STYLES[method]
            axis.plot(method_df["L"], method_df["speedup"], marker="o", linewidth=2, **style)
        axis.set_title(f"d = {d}")
        axis.set_xlabel("Sequence length L")
        axis.set_ylabel("Speedup vs cpu_baseline")
        axis.grid(alpha=0.3)
        axis.legend()

    for axis in axes[len(ds):]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(SPEEDUP_PLOT, dpi=200)
    plt.close(fig)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark CSV: {CSV_PATH}")
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    plot_runtime(df)
    plot_speedup(df)
    print(f"Wrote {RUNTIME_PLOT}")
    print(f"Wrote {SPEEDUP_PLOT}")


if __name__ == "__main__":
    main()
