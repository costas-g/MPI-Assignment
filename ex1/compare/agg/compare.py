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
data_root = os.path.join(ROOT, "data")
os.makedirs(data_root, exist_ok=True)

runs_root = os.path.join(data_root, "runs")
os.makedirs(runs_root, exist_ok=True)

PLOTS_DIR = os.path.join(data_root, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Input CSV  put your files here in the same folder as this script. #for now we have the openMP  vs  pthread
PTHREAD_CSV = os.path.join(ROOT, "pth_poly_results_stats.csv")
OPENMP_CSV  = os.path.join(ROOT, "omp_poly_results_stats.csv")
MPI_CSV     = os.path.join(ROOT, "mpi_poly_results_stats.csv")   
MPI1_CSV    = os.path.join(ROOT, "mpi_poly_results_stats_1node.csv")   


OUT_CSV = os.path.join(runs_root, "speedup_compare.csv")


def parse_speedup_csv(csv_path: str, backend: str) -> pd.DataFrame:
    """
    Reads a CSV with columns: degree, workers, speedup
    Returns a DataFrame with: backend, degree, workers, speedup
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalize column names just in case (Degree vs degree etc.)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"degree", "workers", "speedup"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {csv_path}. Found: {list(df.columns)}")

    out = df[["degree", "workers", "speedup"]].copy()
    out["backend"] = backend

    # enforce types
    out["degree"] = out["degree"].astype(int)
    out["workers"] = out["workers"].astype(int)
    out["speedup"] = out["speedup"].astype(float)

    return out.sort_values(["degree", "workers"]).reset_index(drop=True)



# ---------- PLOTTING ----------

def plot_speedup_compare(df_all: pd.DataFrame) -> None:
    for deg in sorted(df_all["degree"].unique()):
        gdeg = df_all[df_all["degree"] == deg].copy()
        if gdeg.empty:
            continue

        fig, ax = plt.subplots()

        for backend in sorted(gdeg["backend"].unique()):
            gb = gdeg[gdeg["backend"] == backend].sort_values("workers")
            label_suffix = " (threads)" if backend == "OpenMP" or backend == "Pthreads" else " (processes)" if backend == "MPI" or backend == "MPI (1 node)" else ""
            ax.plot(gb["workers"].values, gb["speedup"].values, "o--", label=backend + label_suffix)

        ax.set_xlabel("Workers")
        ax.set_ylabel("Speedup")
        ax.set_title(f"[Poly Mult] Model Comparison - Speedup vs Workers (Degree={deg:,})")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()

        out_path = os.path.join(PLOTS_DIR, f"compare_speedup_deg{deg//1000}K.svg")
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved: {out_path}")
        
        
# ---------- MAIN ----------

def main():
    
    df_openmp = parse_speedup_csv(OPENMP_CSV, "OpenMP")
    df_pth    = parse_speedup_csv(PTHREAD_CSV, "Pthreads")
    df_mpi    = parse_speedup_csv(MPI_CSV, "MPI")
    df_mpi1   = parse_speedup_csv(MPI1_CSV, "MPI (1 node)")

    df_all = pd.concat([df_openmp, df_pth, df_mpi, df_mpi1], ignore_index=True)
    # df_all = pd.concat([df_openmp, df_pth, df_mpi], ignore_index=True)
    # df_all = pd.concat([df_openmp, df_mpi], ignore_index=True)
    df_all.to_csv(OUT_CSV, index=False)
    print(f"[INFO] Saved combined CSV: {OUT_CSV}")

    plot_speedup_compare(df_all)   


if __name__ == "__main__":
    main()
