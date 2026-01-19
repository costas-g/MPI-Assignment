#!/usr/bin/env python3
"""
Benchmark driver for the polynomial multiplication experiments.

Runs the main executable for different (degree, threads) combinations,
records generation, serial, and parallel times, computes speedup and
efficiency, and saves:

Usage:
  python benchmark.py [-s]

Arguments:
  -s         Skip experiments; use existing CSV.

Typical use:
  - Run once without -s to generate data.
  - Re-run with -s to rebuild plots/tables only.
"""
#!/usr/bin/env python3
import os
import re
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from datetime import datetime
import argparse

# ---------------- CONFIG ----------------
ROOT = os.path.abspath(os.path.dirname(__file__))
BIN_DIR = os.path.join(ROOT, "bin")

parser = argparse.ArgumentParser(description="Run polynomial multiplication (serial vs parallel)")
parser.add_argument("-s", "--skip-experiments", action="store_true", help="Skip running experiments and use saved CSV")

args = parser.parse_args()


# Output paths
data_root = os.path.join(ROOT, "data_samenode")
os.makedirs(data_root, exist_ok=True)

runs_root = os.path.join(data_root, "runs")
os.makedirs(runs_root, exist_ok=True)

PLOTS_DIR = os.path.join(data_root, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Input CSV  (schange the name if needed)
STATS_CSV = os.path.join(runs_root, "mpi_poly_results_stats_samenode.csv")


def parse_data_csv(csv_path: str, backend: str) -> pd.DataFrame:
    """
    Reads a CSV with columns: degree, processes, parallel mean , serial mean , speedup, serial_std
    Returns a DataFrame with: backend, degree, processes, parallel mean , serial mean , speedup, serial_std
    
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalize column names just in case (Degree vs degree etc.)
    df.columns = [c.strip().lower() for c in df.columns]

    # required = {"degree", "processes", "serial_mean", "serial_std","parallel_mean", "speedup"}
    required = {"degree", "processes", "serial_min", "parallel_mean", "speedup_calc", "send_time", "compute_time", "reduce_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {csv_path}. Found: {list(df.columns)}")

    # out = df[["degree", "processes", "serial_mean", "serial_std", "parallel_mean", "speedup"]].copy()
    out = df[["degree", "processes", "serial_min", "parallel_mean", "speedup_calc", "send_time", "compute_time", "reduce_time"]].copy()
    out["backend"] = backend

    # enforce types
    out["degree"] = out["degree"].astype(int)
    out["processes"] = out["processes"].astype(int)
    # out["serial_mean"] = out["serial_mean"].astype(float)
    out["serial_min"] = out["serial_min"].astype(float)
    # out["serial_std"] = out["serial_std"].astype(float)
    out["parallel_mean"] = out["parallel_mean"].astype(float)
    # out["speedup"] = out["speedup"].astype(float)
    out["speedup_calc"] = out["speedup_calc"].astype(float)
    out["send_time"] = out["send_time"].astype(float)
    out["compute_time"] = out["compute_time"].astype(float)
    out["reduce_time"] = out["reduce_time"].astype(float)
    

    return out.sort_values(["degree", "processes"]).reset_index(drop=True)

# ---------- PLOTTING ----------

def plot_stats(stats_df):
    """
    Plot parallel time vs processes for each degree.
    - Also draw the serial mean as a horizontal dashed line.
    - One PDF per degree under data/plots/.
    """
    for degree, group in stats_df.groupby("degree"):
        group = group.sort_values("processes")

        threads = group["processes"].values
        t_mean  = group["parallel_mean"].values
        #t_min   = group["parallel_min"].values
        #t_max   = group["parallel_max"].values
        #t_std   = group["parallel_std"].values  # <-- use std for error bars

        #yerr_min = t_mean - t_min
        #yerr_max = t_max - t_mean
        #yerr = np.vstack([yerr_min, yerr_max])

        serial_min_min = group["serial_min"].min()
        # serial_mean = group["serial_mean"].iloc[0]
        # serial_std  = group["serial_std"].iloc[0]

        # --------------- Plot send / compute / reduce times ---------------
        width = 0.25  # width of each bar
        x = np.arange(len(threads))  # positions for the groups
        fig, ax = plt.subplots()

        ax.axhline(serial_min_min, linestyle="-", color="green", label=f"Serial time ({serial_min_min:.4f}s)")

        if degree < 10**4:
            ax.set_yscale("log")
            ax.plot(threads, group["send_time"], "o--", label="Send time")
            ax.plot(threads, group["compute_time"], "s--", label="Compute time")
            ax.plot(threads, group["reduce_time"], "d--", label="Reduce time")
            ax.plot(threads,t_mean, "o--", label=f"Parallel time total",)
        else:
            ax.bar(threads, group["send_time"], label="Send")
            ax.bar(threads, group["compute_time"], bottom=group["send_time"], label="Compute")
            ax.bar(threads, group["reduce_time"], bottom=group["send_time"]+group["compute_time"], label="Reduce")
            ax.plot(threads,t_mean, "o--", label=f"Parallel time total",)
        
        # ax.bar(x - width, group["send_time"],    width=width, label="Send")
        # ax.bar(x,         group["compute_time"], width=width, label="Compute")
        # ax.bar(x + width, group["reduce_time"],  width=width, label="Reduce")

        # ax.set_xticks(x)
        # ax.set_xticklabels(threads)  # show actual process numbers

        

        ax.set_xlabel("processes")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"[MPI Poly mult 1 node] Times breakdown vs processes (degree={degree:,})")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()

        out_breakdown = os.path.join(PLOTS_DIR, f"mpi_poly_breakdown_deg{degree//1000}K.svg")
        fig.savefig(out_breakdown, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved breakdown plot: {out_breakdown}")
        

        # --------------- Plot total time ---------------
        fig, ax = plt.subplots()

        ax.errorbar(
            threads,
            t_mean,
            #yerr=t_std, # <-- error bars are std,
            fmt="o--",
            capsize=3,
            label=f"Parallel (degree={degree:,})",
        )

        ax.axhline(
            serial_min_min,
            linestyle="-",
            color="green",
            label=f"Serial time ({serial_min_min:.4f}s)"
        )
        '''
        # Add shaded region for std not shown cause very small std so not needed3
        ax.fill_between(
            threads, 
            serial_mean - serial_std, 
            serial_mean + serial_std, 
            color="green", 
            alpha=0.2, 
            label=f"Serial +- 1 std"
        )'''

        ax.set_xlabel("processes")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"[MPI Poly mult] Execution Time vs processes (degree={degree:,})")
        # ax.set_yscale("log")  # log scale like the barrier script
        ax.grid(which="major", linestyle="--", alpha=0.7)
        ax.grid(which="minor", linestyle="--", alpha=0.3)
        ax.legend()

        fig.tight_layout()

        out_path = os.path.join(PLOTS_DIR, f"mpi_poly_time_vs_processes_deg{degree//1000}K.svg")
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        plt.close(fig)

        print(f"[INFO] Saved plot: {out_path}")
        

        # --------------- Speedup plot ---------------
        speedup = [serial_min_min / t for t in t_mean] # group["speedup"].values

        fig, ax = plt.subplots()
        ax.plot(threads, speedup, "o--", label="Speedup (serial_mean / parallel_mean)")
        ax.set_xlabel("processes")
        ax.set_ylabel("Speedup")
        ax.set_title(f"[MPI Poly mult 1 node] Speedup vs processes (degree={degree:,})")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()

        out_speed = os.path.join(PLOTS_DIR, f"mpi_poly_speedup_deg{degree//1000}K.svg")
        fig.savefig(out_speed, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved plot: {out_speed}")


    # -------------- Send & Reduce time vs Degree --------------
    # filter constant process count
    p = 4
    df4 = stats_df[stats_df["processes"] == p].sort_values("degree")
    degrees = df4["degree"].values

    fig, ax = plt.subplots()

    ax.plot(degrees, df4["send_time"], "o--", label="Send time")
    ax.plot(degrees, df4["compute_time"], "s--", label="Compute time")
    ax.plot(degrees, df4["reduce_time"], "d--", label="Reduce time")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("degree")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"[MPI Poly mult 1 node] Times vs degree (processes={p})")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    fig.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"mpi_poly_times_v_degree_p{p}.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved plot: {out_path}")

        
# ---------- MAIN ----------

def main():
    
    stats_df = parse_data_csv(STATS_CSV,"polynomial")
    
    plot_stats(stats_df)   


if __name__ == "__main__":
    main()
