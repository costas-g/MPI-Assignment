#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------- CONFIG ----------------

ROOT = os.path.abspath(os.path.dirname(__file__))

DATA_DIR  = os.path.join(ROOT, "data")
RUNS_DIR  = os.path.join(DATA_DIR, "runs")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

MEANS_CSV_PATH = os.path.join(RUNS_DIR, "matrix_results_means.csv")


def make_plots(means_csv: str) -> None:
    df = pd.read_csv(means_csv)

    # Keep only rows with valid metrics
    df_ok = df[df["n_ok"] > 0].copy()
    
    
    # 2) CSR mult scaling vs threads 
    for N in sorted(df_ok["matrix_size"].unique()):
        subN = df_ok[df_ok["matrix_size"] == N].copy()
        reps = sorted(subN["sparsity"].unique(), reverse=True)#[:6]

        fig, ax = plt.subplots()
        for sp in reps:
            g = subN[subN["sparsity"] == sp].groupby("processes")["csr_speedup_s_mean"].mean().reset_index().sort_values("processes")
            ax.plot(g["processes"], g["csr_speedup_s_mean"], "o--", label=f"sp={sp:.2f}")

        ax.set_xlabel("Processes")
        ax.set_ylabel("Speedup (CSR mult)")
        ax.set_title(f"CSR mult speedup vs processes (N={N:,}) ")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"csr_mult_speedup_N{N//1000}K.svg")
        fig.savefig(out, format="svg", bbox_inches="tight")
        plt.close(fig)

    # 3) Dense mult scaling vs threads 
    for N in sorted(df_ok["matrix_size"].unique()):
        subN = df_ok[df_ok["matrix_size"] == N].copy()
        reps = sorted(subN["sparsity"].unique(), reverse=True)#[:6]

        fig, ax = plt.subplots()
        for sp in reps:
            g = subN[subN["sparsity"] == sp].groupby("processes")["dense_speedup_s_mean"].mean().reset_index().sort_values("processes")
            ax.plot(g["processes"], g["dense_speedup_s_mean"], "o--", label=f"sp={sp:.2f}")

        ax.set_xlabel("Processes")
        ax.set_ylabel("Speedup (Dense mult)")
        ax.set_title(f"Dense mult speedup vs processes (N={N:,}) ")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"dense_mult_speedup_N{N//1000}K.svg")
        fig.savefig(out, format="svg", bbox_inches="tight")
        plt.close(fig)

    df_ok["ratio_dense_over_csr_serial"] = df_ok["dense_mult_serial_s_mean"] / df_ok["csr_mult_serial_s_mean"]
    df_ok["ratio_dense_over_csr_parallel"] = df_ok["dense_mult_parallel_s_mean"] / df_ok["csr_mult_parallel_s_mean"]
    
    # 4) CSR vs Dense ratio vs sparsity (t=1 and t=max threads) 
    for N in sorted(df_ok["matrix_size"].unique()):
        subN = df_ok[df_ok["matrix_size"] == N].copy()
        maxT = int(subN["processes"].max())

        #row["ratio_dense_over_csr_serial_per_mult"] = (row["dense_mult_serial_s_mean"] / k) / (row["csr_mult_serial_s_mean"] / k)
        
        serial = subN[subN["processes"] == 1].groupby("sparsity")["ratio_dense_over_csr_serial"].mean().reset_index().sort_values("sparsity")
        par_4    = subN[subN["processes"] == 4].groupby("sparsity")["ratio_dense_over_csr_parallel"].mean().reset_index().sort_values("sparsity")
        par_max    = subN[subN["processes"] == maxT].groupby("sparsity")["ratio_dense_over_csr_parallel"].mean().reset_index().sort_values("sparsity")

        fig, ax = plt.subplots()
        ax.axhline(1.0, color="black", linestyle="-", alpha=0.6) # baseline
        ax.plot(serial["sparsity"], serial["ratio_dense_over_csr_serial"], "o--", label="t=1")
        ax.plot(par_4["sparsity"], par_4["ratio_dense_over_csr_parallel"], "o--", label="t=4")
        ax.plot(par_max["sparsity"], par_max["ratio_dense_over_csr_parallel"], "o--", label=f"t={maxT}")
        

        ax.set_xlabel("Sparsity")
        ax.set_ylabel("Dense / CSR  (>1 => CSR faster)")

        ax.set_yscale("log", base=10)
        ymax = max(
            serial["ratio_dense_over_csr_serial"].max(),
            par_max["ratio_dense_over_csr_parallel"].max(),
            par_4["ratio_dense_over_csr_parallel"].max(),
        )
        ax.set_ylim(top=10 ** np.ceil(np.log10(ymax)))
        # Gridlines
        ax.grid(True, linestyle="-", alpha=0.6)
        ax.grid(True, which="major", axis="y", linestyle="-", alpha=0.6)
        ax.grid(True, which="minor", axis="y", linestyle="--", alpha=0.6)
        ax.grid(True, which="minor", axis="x", linestyle="--", alpha=0.3)
        
        ax.set_title(f"CSR vs Dense ratio vs sparsity (N={N:,})")

        ax.legend()
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"csr_vs_dense_ratio_N{N//1000}K.svg")
        fig.savefig(out, format="svg", bbox_inches="tight")
        plt.close(fig)

    print(f"[INFO] Plots saved in: {PLOTS_DIR}")


def main():
    
        make_plots(MEANS_CSV_PATH)
        
if __name__ == "__main__":
    main()
