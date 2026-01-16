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

    required = {"degree", "processes", "serial_mean", "serial_std","parallel_mean", "speedup"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {csv_path}. Found: {list(df.columns)}")

    out = df[["degree", "processes","serial_mean","serial_std", "parallel_mean", "speedup"]].copy()
    out["backend"] = backend

    # enforce types
    out["degree"] = out["degree"].astype(int)
    out["processes"] = out["processes"].astype(int)
    out["serial_mean"] = out["serial_mean"].astype(float)
    out["serial_std"] = out["serial_std"].astype(float)
    out["parallel_mean"] = out["parallel_mean"].astype(float)
    out["speedup"] = out["speedup"].astype(float)
    

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

        serial_mean = group["serial_mean"].iloc[0]
        serial_std  = group["serial_std"].iloc[0]

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
            serial_mean,
            linestyle="-",
            color="green",
            label=f"Serial mean ({serial_mean:.4f}s)"
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
        
        # ---- Speedup plot ----
        speedup = [serial_mean / t for t in t_mean] # group["speedup"].values

        fig, ax = plt.subplots()
        ax.plot(threads, speedup, "o--", label="Speedup (serial_mean / parallel_mean)")
        ax.set_xlabel("processes")
        ax.set_ylabel("Speedup")
        ax.set_title(f"[MPI Poly mult] Speedup vs processes (degree={degree:,})")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()

        out_speed = os.path.join(PLOTS_DIR, f"mpi_poly_speedup_deg{degree//1000}K.svg")
        fig.savefig(out_speed, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved plot: {out_speed}")
        
# ---------- MAIN ----------

def main():
    
    stats_df = parse_data_csv(STATS_CSV,"polynomial")
    
    plot_stats(stats_df)   


if __name__ == "__main__":
    main()
