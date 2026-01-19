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
    
    # 1) CSR build vs sparsity
    # for N in sorted(df_ok["matrix_size"].unique()):
    # subN = df_ok[df_ok["matrix_size"] == N].copy()
    subN = df_ok.copy()
    reps = sorted(subN["matrix_size"].unique(), reverse=True)

    fig, ax = plt.subplots()
    for nn in reps:
        g = subN[subN["matrix_size"] == nn].groupby("sparsity")["csr_build_serial_s_mean"].mean().reset_index().sort_values("sparsity")
        ax.plot(g["sparsity"], g["csr_build_serial_s_mean"], "o--", label=f"matrix order={nn//1000}K")

    ax.set_yscale("log")
    ax.set_ylim(10**-3, 10**0)

    ax.set_xlabel("Sparsity")
    ax.set_ylabel("CSR Build time (s)")
    ax.set_title(f"CSR build time vs sparsity")
    # ax.set_title(f"CSR mult speedup vs sparsity (N={N:,}) ")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    # out = os.path.join(PLOTS_DIR, f"csr_build_v_sparsity_N{N//1000}K.svg")
    out = os.path.join(PLOTS_DIR, f"csr_build_v_sparsity.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved plot: {out}")


    # 1.2) CSR send vs sparsity
    # for N in sorted(df_ok["matrix_size"].unique()):
    # subN = df_ok[df_ok["matrix_size"] == N].copy()
    subN = df_ok.copy()
    reps = sorted(subN["matrix_size"].unique(), reverse=True)

    fig, ax = plt.subplots()
    for nn in reps:
        g = subN[subN["matrix_size"] == nn].groupby("sparsity")["data_send_time_s_mean"].mean().reset_index().sort_values("sparsity")
        ax.plot(g["sparsity"], g["data_send_time_s_mean"], "o--", label=f"matrix order={nn//1000}K")

    ax.set_yscale("log")
    ax.set_ylim(10**-3, 10**1)

    ax.set_xlabel("Sparsity")
    ax.set_ylabel("CSR Send time (s)")
    ax.set_title(f"CSR Send time vs sparsity")
    # ax.set_title(f"CSR mult speedup vs sparsity (N={N:,}) ")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    # out = os.path.join(PLOTS_DIR, f"csr_build_v_sparsity_N{N//1000}K.svg")
    out = os.path.join(PLOTS_DIR, f"csr_send_v_sparsity.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved plot: {out}")

    # 1.2.5) CSR send vs procs
    # for N in sorted(df_ok["matrix_size"].unique()):
    # subN = df_ok[df_ok["sparsity"] == 0.9].copy()
    subN = df_ok.copy()
    reps = sorted(subN["matrix_size"].unique(), reverse=True)

    fig, ax = plt.subplots()
    for nn in reps:
        g = subN[subN["matrix_size"] == nn].groupby("processes")["data_send_time_s_mean"].mean().reset_index().sort_values("processes")
        ax.plot(g["processes"], g["data_send_time_s_mean"], "o--", label=f"matrix order={nn//1000}K")

    ax.set_yscale("log")
    ax.set_ylim(10**-3, 10**1)

    ax.set_xlabel("Processes")
    ax.set_ylabel("CSR Send time (s)")
    ax.set_title(f"CSR Send time vs processes")
    # ax.set_title(f"CSR mult speedup vs sparsity (N={N:,}) ")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    # out = os.path.join(PLOTS_DIR, f"csr_build_v_sparsity_N{N//1000}K.svg")
    out = os.path.join(PLOTS_DIR, f"csr_send_v_procs.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved plot: {out}")
    

    # 1.3) CSR compute vs sparsity (mults = 20) (procs = 4)
    # for N in sorted(df_ok["matrix_size"].unique()):
    mult_reps = 20
    num_procs = 4
    # subN = df_ok[(df_ok["num_mults"] == mult_reps) & (df_ok["processes"] == num_procs)].copy()
    subN = df_ok.copy()
    reps = sorted(subN["matrix_size"].unique(), reverse=True)

    fig, ax = plt.subplots()
    for nn in reps:
        g = subN[subN["matrix_size"] == nn].groupby("sparsity")["parallel_compute_s_mean"].mean().reset_index().sort_values("sparsity")
        ax.plot(g["sparsity"], g["parallel_compute_s_mean"], "o--", label=f"matrix order={nn//1000}K")

    ax.set_yscale("log")
    ax.set_ylim(10**-3, 10**0)

    ax.set_xlabel("Sparsity")
    ax.set_ylabel("CSR Compute time (s)")
    ax.set_title(f"CSR Compute time vs sparsity")# (mults={mult_reps}, procs={num_procs})")
    # ax.set_title(f"CSR mult speedup vs sparsity (N={N:,}) ")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    # out = os.path.join(PLOTS_DIR, f"csr_build_v_sparsity_N{N//1000}K.svg")
    out = os.path.join(PLOTS_DIR, f"csr_compute_v_sparsity.svg")#_m{mult_reps}_p{num_procs}.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved plot: {out}")

    # 1.4) CSR compute vs mults (sp = 0.9) (procs = 4)
    # for N in sorted(df_ok["matrix_size"].unique()):
    sp = 0.95
    num_procs = 4
    # subN = df_ok[(df_ok["sparsity"] == sp) & (df_ok["processes"] == num_procs)].copy()
    subN = df_ok.copy()
    reps = sorted(subN["matrix_size"].unique(), reverse=True)

    fig, ax = plt.subplots()
    for nn in reps:
        g = subN[subN["matrix_size"] == nn].groupby("num_mults")["parallel_compute_s_mean"].mean().reset_index().sort_values("num_mults")
        ax.plot(g["num_mults"], g["parallel_compute_s_mean"], "o--", label=f"matrix order={nn//1000}K")

    ax.set_yscale("log")
    # ax.set_ylim(10**-2, 10**0)

    ax.set_xlabel("Multiplications")
    ax.set_ylabel("CSR Compute time (s)")
    ax.set_title(f"CSR Compute time vs multiplications")# (sparsity={sp}, procs={num_procs})")
    # ax.set_title(f"CSR mult speedup vs sparsity (N={N:,}) ")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    # out = os.path.join(PLOTS_DIR, f"csr_build_v_sparsity_N{N//1000}K.svg")
    out = os.path.join(PLOTS_DIR, f"csr_compute_v_mults.svg")#_sp{sp:.2f}_p{num_procs}.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved plot: {out}")

    # 1.5) CSR compute vs procs (sp = 0.9) (mults = 20)
    # for N in sorted(df_ok["matrix_size"].unique()):
    # sp = 0.95
    # mult_reps = 20
    # subN = df_ok[(df_ok["sparsity"] == sp) & (df_ok["num_mults"] == mult_reps)].copy()
    subN = df_ok.copy()
    reps = sorted(subN["matrix_size"].unique(), reverse=True)

    fig, ax = plt.subplots()
    for nn in reps:
        g = subN[subN["matrix_size"] == nn].groupby("processes")["parallel_compute_s_mean"].mean().reset_index().sort_values("processes")
        ax.plot(g["processes"], g["parallel_compute_s_mean"], "o--", label=f"matrix order={nn//1000}K")

    ax.set_yscale("log")
    # ax.set_ylim(10**-2, 10**0)

    ax.set_xlabel("processes")
    ax.set_ylabel("CSR Compute time (s)")
    ax.set_title(f"CSR Compute time vs processes")# (sparsity={sp}, mults={mult_reps})")
    # ax.set_title(f"CSR mult speedup vs sparsity (N={N:,}) ")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    # out = os.path.join(PLOTS_DIR, f"csr_build_v_sparsity_N{N//1000}K.svg")
    out = os.path.join(PLOTS_DIR, f"csr_compute_v_procs.svg")#_sp{sp:.2f}_m{mult_reps}.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved plot: {out}")


    # 2) CSR mult total speedup vs threads 
    for N in sorted(df_ok["matrix_size"].unique()):
        subN = df_ok[df_ok["matrix_size"] == N].copy()
        reps = sorted(subN["sparsity"].unique(), reverse=True)#[:6]

        fig, ax = plt.subplots()
        for sp in reps:
            # g = subN[subN["sparsity"] == sp].groupby("processes")["csr_speedup_s_mean"].mean().reset_index().sort_values("processes")
            g1 = subN[subN["sparsity"] == sp].copy()

            g1["serial_candidate"] = g1["csr_mult_serial_s_mean"]
            mask = g1["processes"] == 1
            g1.loc[mask, "serial_candidate"] = g1.loc[mask, "csr_mult_parallel_s_mean"]

            g1["serial_min"] = (
                g1.groupby("num_mults")["serial_candidate"].transform("min")
            )

            g1["real_speedup"] = g1["serial_min"] / g1["csr_mult_parallel_s_mean"]

            g2 = (
                g1
                .groupby(["processes"])["real_speedup"]
                .mean()
                .reset_index(name="real_speedup_mean")
            )
            # ax.plot(g["processes"], g["csr_speedup_s_mean"], "o--", label=f"sp={sp:.2f}")
            ax.plot(g["processes"], g2["real_speedup_mean"], "o--", label=f"sp={sp:.2f}")

        ax.set_xlabel("Processes")
        ax.set_ylabel("Total Speedup (CSR SpMV)")
        ax.set_title(f"CSR SpMV Total Speedup vs Processes (N={N:,}) ")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"csr_mv_total_speedup_N{N//1000}K.svg")
        fig.savefig(out, format="svg", bbox_inches="tight")
        plt.close(fig)

        print(f"[INFO] Saved plot: {out}")

    
    # 2.2) CSR mult compute speedup vs threads 
    for N in sorted(df_ok["matrix_size"].unique()):
        subN = df_ok[df_ok["matrix_size"] == N].copy()
        reps = sorted(subN["sparsity"].unique(), reverse=True)#[:6]

        fig, ax = plt.subplots()
        for sp in reps:
            g = subN[subN["sparsity"] == sp].groupby("processes")["csr_speedup_s_mean"].mean().reset_index().sort_values("processes")
            ax.plot(g["processes"], g["csr_speedup_s_mean"], "o--", label=f"sp={sp:.2f}")

        ax.set_xlabel("Processes")
        ax.set_ylabel("Compute Speedup (CSR SpMV)")
        ax.set_title(f"[CSR SpMV] Compute Speedup vs Processes (N={N:,}) ")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"csr_mv_compute_speedup_N{N//1000}K.svg")
        fig.savefig(out, format="svg", bbox_inches="tight")
        plt.close(fig)

        print(f"[INFO] Saved plot: {out}")


    # 3) Dense mult scaling vs threads 
    for N in sorted(df_ok["matrix_size"].unique()):
        subN = df_ok[df_ok["matrix_size"] == N].copy()
        reps = sorted(subN["sparsity"].unique(), reverse=True)#[:6]

        fig, ax = plt.subplots()

        for sp in reps:
            df = subN[subN["sparsity"] == sp].copy()

            # initialize column with NaN
            df["dense_compute_time"] = np.nan

            # fill only for processes == 1
            # mask = df["processes"] == 1
            df["dense_compute_time"] = df["dense_mult_serial_s_mean"] / df["dense_speedup_s_mean"]

            df["dense_compute_time_p1"] = np.nan  # initialize
            mask = df["processes"] == 1
            df.loc[mask, "dense_compute_time_p1"] = df.loc[mask, "dense_compute_time"]

            # real serial min
            df["real_serial_min"] = df.groupby("num_mults")[["dense_mult_serial_s_mean", "dense_compute_time_p1"]].transform("min").min(axis=1)


            # # serial candidate
            # df["serial_candidate"] = df["dense_mult_serial_s_mean"]
            # mask = df["processes"] == 1
            # df.loc[mask, "serial_candidate"] = df.loc[mask, "dense_mult_parallel_s_mean"]

            # # serial candidate
            # df["serial_candidate"] = df["dense_mult_serial_s_mean"]
            # mask = df["processes"] == 1
            # df.loc[mask, "serial_candidate"] = df.loc[mask, ["dense_mult_serial_s_mean", "dense_mult_parallel_s_mean"]].min(axis=1)

            # # true serial reference
            # df["serial_min"] = (
            #     df.groupby("num_mults")["serial_candidate"].transform("min")
            # )

            # Step 2: compute speedup per row (no groupby)
            df["compute_speedup"] = df["real_serial_min"] / df["dense_compute_time"] 

            # # compute speedup
            # df["compute_speedup"] = df["serial_min"] / (df["dense_mult_serial_s_mean"]  / df["dense_speedup_s_mean"])

            g = (
                df.groupby("processes")["compute_speedup"]
                .mean()
                .reset_index()
                .sort_values("processes")
            )

            ax.plot(g["processes"], g["compute_speedup"], "o--", label=f"sp={sp:.2f}")

        # for sp in reps:
        #     g = subN[subN["sparsity"] == sp].groupby("processes")["dense_speedup_s_mean"].mean().reset_index().sort_values("processes")
        #     ax.plot(g["processes"], g["dense_speedup_s_mean"], "o--", label=f"sp={sp:.2f}")

        ax.set_xlabel("Processes")
        ax.set_ylabel("Compute Speedup (Dense MV)")
        ax.set_title(f"Dense MV Compute speedup vs processes (N={N:,}) ")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"dense_mv_compute_speedup_N{N//1000}K.svg")
        fig.savefig(out, format="svg", bbox_inches="tight")
        plt.close(fig)

        print(f"[INFO] Saved plot: {out}")

    
    # 3.2) FIXED Dense mult scaling vs threads 
    for N in sorted(df_ok["matrix_size"].unique()):
        subN = df_ok[df_ok["matrix_size"] == N].copy()
        reps = sorted(subN["sparsity"].unique(), reverse=True)

        fig, ax = plt.subplots()

        for sp in reps:
            df = subN[subN["sparsity"] == sp].copy()

            # serial candidate
            df["serial_candidate"] = df["dense_mult_serial_s_mean"]
            mask = df["processes"] == 1
            df.loc[mask, "serial_candidate"] = df.loc[mask, "dense_mult_parallel_s_mean"]

            # true serial reference
            df["serial_min"] = (
                df.groupby("num_mults")["serial_candidate"].transform("min")
            )

            # real speedup
            df["real_speedup"] = df["serial_min"] / df["dense_mult_parallel_s_mean"]

            g = (
                df.groupby("processes")["real_speedup"]
                .mean()
                .reset_index()
                .sort_values("processes")
            )

            ax.plot(g["processes"], g["real_speedup"], "o--", label=f"sp={sp:.2f}")

        ax.set_xlabel("Processes")
        ax.set_ylabel("Total Speedup (Dense MV)")
        ax.set_title(f"Dense MV Total speedup vs processes (N={N:,})")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()

        out = os.path.join(PLOTS_DIR, f"dense_mv_total_speedup_N{N//1000}K.svg")
        fig.savefig(out, format="svg", bbox_inches="tight")
        plt.close(fig)

        print(f"[INFO] Saved plot: {out}")

        
    # 4) CSR vs Dense ratio vs sparsity (t=1 and t=max threads) 
    df_ok["ratio_dense_over_csr_serial"] = df_ok["dense_mult_serial_s_mean"] / df_ok["csr_mult_serial_s_mean"]
    df_ok["ratio_dense_over_csr_parallel"] = df_ok["dense_mult_parallel_s_mean"] / df_ok["csr_mult_parallel_s_mean"]

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

        print(f"[INFO] Saved plot: {out}")

    print(f"[INFO] Plots saved in: {PLOTS_DIR}")


def main():
    
        make_plots(MEANS_CSV_PATH)
        
if __name__ == "__main__":
    main()
