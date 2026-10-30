import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

def iqr(series):
    return np.percentile(series, 75) - np.percentile(series, 25)

df_a = pd.read_csv("results/raw_partA_dense.csv")
summary_a = df_a.groupby(["n", "method"], as_index=False).agg(
    repetitions=("rep", "count"),
    median_time_s=("kernel_time_s", "median"),
    iqr_time_s=("kernel_time_s", iqr),
    structure_mem_mb=("structure_mem_mb", "first"),
    median_rss_mb=("peak_rss_mb", "median")
)

base_times = summary_a[summary_a["method"] == "dense_ijk"].set_index("n")["median_time_s"]
summary_a["speedup_vs_ijk"] = summary_a.apply(
    lambda row: base_times[row["n"]] / row["median_time_s"], axis=1
)
summary_a.to_csv("results/summary_partA_dense.csv", index=False)
print("\n=== SUMMARY PART A (Median & IQR) ===")
print(summary_a.to_string(index=False))

plt.figure(figsize=(8, 5))
for method, label, marker in [
    ("dense_ijk", "Baseline (dense_ijk)", "o"),
    ("dense_ikj", "Loop Reordering (dense_ikj)", "s"),
    ("dense_tiled", "Cache Blocking B=64 (dense_tiled)", "^")
]:
    sub = summary_a[summary_a["method"] == method].sort_values("n")
    plt.errorbar(
        sub["n"], sub["median_time_s"], yerr=sub["iqr_time_s"] / 2.0,
        label=label, marker=marker, capsize=4, linewidth=2
    )
plt.yscale("log")
plt.xlabel("Matrix Dimension $n$ ($n \\times n$)", fontsize=11)
plt.ylabel("Kernel Execution Time [s] (Median ± IQR/2, log scale)", fontsize=11)
plt.title("Part A: Dense Matrix Multiplication Runtime vs. Size $n$", fontsize=12)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig1_dense_runtime.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
for method, label, marker in [
    ("dense_ijk", "Baseline (dense_ijk)", "o"),
    ("dense_ikj", "Loop Reordering (dense_ikj)", "s"),
    ("dense_tiled", "Cache Blocking B=64 (dense_tiled)", "^")
]:
    sub = summary_a[summary_a["method"] == method].sort_values("n")
    plt.plot(sub["n"], sub["speedup_vs_ijk"], label=label, marker=marker, linewidth=2)
plt.xlabel("Matrix Dimension $n$ ($n \\times n$)", fontsize=11)
plt.ylabel("Speedup over Baseline ($T_{ijk} / T_{opt}$)", fontsize=11)
plt.title("Part A: Speedup of Dense Optimisations Relative to Baseline", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig2_dense_speedup.png", dpi=300)
plt.close()

df_b = pd.read_csv("results/raw_partB_sparse.csv")
summary_b = df_b.groupby(["n", "density", "sparsity", "method", "format"], as_index=False).agg(
    repetitions=("rep", "count"),
    median_conv_s=("conversion_time_s", "median"),
    median_kernel_s=("kernel_time_s", "median"),
    iqr_kernel_s=("kernel_time_s", iqr),
    median_total_s=("total_time_s", "median"),
    iqr_total_s=("total_time_s", iqr),
    nnz_A=("nnz_A", "first"),
    nnz_B=("nnz_B", "first"),
    nnz_C=("nnz_C", "first"),
    structure_mem_mb=("structure_mem_mb", "first"),
    median_rss_mb=("peak_rss_mb", "median")
)
summary_b["output_density_pct"] = (summary_b["nnz_C"] / (summary_b["n"] ** 2)) * 100.0
summary_b.to_csv("results/summary_partB_sparse.csv", index=False)
print("\n=== SUMMARY PART B (Median & IQR) ===")
print(summary_b.to_string(index=False))

sub_1024 = summary_b[summary_b["n"] == 1024].sort_values("density")
dense_1024 = sub_1024[sub_1024["method"] == "dense_ikj"]
sparse_1024 = sub_1024[sub_1024["method"] == "sparse_csr"]

plt.figure(figsize=(8, 5))
plt.plot(dense_1024["density"] * 100, dense_1024["median_kernel_s"], label="Dense Optimised (dense_ikj)", marker="s", linewidth=2)
plt.plot(sparse_1024["density"] * 100, sparse_1024["median_kernel_s"], label="Sparse CSR (Kernel Only)", marker="o", linewidth=2)
plt.plot(sparse_1024["density"] * 100, sparse_1024["median_total_s"], label="Sparse CSR (Kernel + Conversion)", marker="^", linestyle="--", linewidth=2)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Input Matrix Density [% Non-Zero Entries] (log scale)", fontsize=11)
plt.ylabel("Execution Time [s] (Median, log scale)", fontsize=11)
plt.title("Part B: Sparse (CSR) vs. Dense Runtime across Density Levels ($n=1024$)", fontsize=12)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig3_sparse_vs_dense_breakeven.png", dpi=300)
plt.close()

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(sparse_1024["density"] * 100, sparse_1024["structure_mem_mb"], color="tab:blue", marker="o", linewidth=2, label="CSR Storage + Accumulator [MB]")
ax1.axhline(y=dense_1024["structure_mem_mb"].iloc[0], color="tab:blue", linestyle=":", linewidth=2, label="Dense Storage (3 Matrices) [MB]")
ax1.set_xlabel("Input Matrix Density [% Non-Zero Entries]", fontsize=11)
ax1.set_ylabel("Data Structure Memory [MB]", color="tab:blue", fontsize=11)
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.grid(True, linestyle="--", alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(sparse_1024["density"] * 100, sparse_1024["output_density_pct"], color="tab:red", marker="s", linestyle="--", linewidth=2, label="Output Product $C$ Density [%]")
ax2.set_ylabel("Output Matrix $C$ Non-Zero Density [%] (Fill-in)", color="tab:red", fontsize=11)
ax2.tick_params(axis="y", labelcolor="tab:red")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
plt.title("Part B: Memory Footprint and Output Fill-in vs. Input Density ($n=1024$)", fontsize=12)
plt.tight_layout()
plt.savefig("figures/fig4_sparse_memory_and_fillin.png", dpi=300)
plt.close()

print("\nPlots saved in figures/ and summary CSVs saved in results/!")
