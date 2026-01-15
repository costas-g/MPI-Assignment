#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import argparse
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------- CONFIG ----------------

ROOT = os.path.abspath(os.path.dirname(__file__))
BIN_DIR = os.path.join(ROOT, "bin")

EXE_PATH = os.path.join(BIN_DIR, "main")
if not os.path.exists(EXE_PATH) and os.path.exists(EXE_PATH + ".exe"):
    EXE_PATH = EXE_PATH + ".exe"

MATRIX_SIZE = [1024, 4096, 8192]

# IMPORTANT: your main expects sparsity as float in [0,1]
SPARSITY = [0.01, 0.1, 0.4, 0.8]  # 0.01, 0.06, ..., 0.96

NUM_MULTS = [1, 5, 10]
PROCESSES   = [1,2,4,8]

DATA_DIR  = os.path.join(ROOT, "data")
EXEC_DIR  = os.path.join(DATA_DIR, "exec")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")

os.makedirs(EXEC_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

MEANS_CSV_PATH = os.path.join(EXEC_DIR, "matrix_results_means.csv")


# ---------------- REGEXES  ----------------
FLOAT_PAT = r"([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"
PREFIX = r"^\s*"  

# Serial CSR build
csr_build_serial = re.compile(PREFIX + r"Serial CSR build time \(s\):\s*" + FLOAT_PAT, re.MULTILINE)


csr_build_parallel = re.compile(PREFIX + r"build\s*:\s*" + FLOAT_PAT, re.MULTILINE)

# Sparse mult time lines are: "Sparse matrix 1x mult Parallel time (s): ..."
csr_mult_parallel = re.compile(PREFIX + r"Sparse matrix \d+x mult Parallel time \(s\):\s*" + FLOAT_PAT, re.MULTILINE)
csr_mult_serial   = re.compile(PREFIX + r"Sparse matrix \d+x mult Serial time \(s\):\s*" + FLOAT_PAT, re.MULTILINE)

# Dense mult time lines are: "Dense matrix 1x mult Parallel time (s): ..."
dense_mult_parallel = re.compile(PREFIX + r"Dense matrix \d+x mult Parallel time \(s\):\s*" + FLOAT_PAT, re.MULTILINE)
dense_mult_serial   = re.compile(PREFIX + r"Dense matrix \d+x mult Serial time \(s\):\s*" + FLOAT_PAT, re.MULTILINE)

# Speedups
dense_csr_speedup_p_regex = re.compile(PREFIX + r"Dense\s*/\s*CSR Speedup in Parallel:\s*" + FLOAT_PAT, re.MULTILINE)
dense_csr_speedup_s_regex = re.compile(PREFIX + r"Dense\s*/\s*CSR Speedup in Serial:\s*" + FLOAT_PAT, re.MULTILINE)

dense_speedup_regex = re.compile(PREFIX + r"Dense MV Speedup:\s*" + FLOAT_PAT, re.MULTILINE)
csr_speedup_regex   = re.compile(PREFIX + r"CSR MV Speedup:\s*" + FLOAT_PAT, re.MULTILINE)


key_fields = [
    "csr_build_serial_s",
    "csr_build_parallel_s",
    "dense_mult_serial_s",
    "dense_mult_parallel_s",
    "csr_mult_serial_s",
    "csr_mult_parallel_s",
    "dense_csr_speedup_p",
    "dense_csr_speedup_s",
    "dense_speedup_s", #final speedup what we need 
    "csr_speedup_s",   # this too is final speedup
]


def _last_float(pat: re.Pattern, text: str) -> Optional[float]:
    m = None
    for m in pat.finditer(text):
        pass
    return float(m.group(1)) if m else None


def parse_output(output: str):
    return {
        "csr_build_serial_s":   _last_float(csr_build_serial, output),
        "csr_build_parallel_s": _last_float(csr_build_parallel, output),

        "dense_mult_serial_s":   _last_float(dense_mult_serial, output),
        "dense_mult_parallel_s": _last_float(dense_mult_parallel, output),

        "csr_mult_serial_s":   _last_float(csr_mult_serial, output),
        "csr_mult_parallel_s": _last_float(csr_mult_parallel, output),

        "dense_csr_speedup_p": _last_float(dense_csr_speedup_p_regex, output),
        "dense_csr_speedup_s": _last_float(dense_csr_speedup_s_regex, output),

        "dense_speedup_s": _last_float(dense_speedup_regex, output),
        "csr_speedup_s":   _last_float(csr_speedup_regex, output),
    }



def run_once(exe_path: str, N: int, sp: float, k: int, t: int) -> Dict[str, Any]:
    cmd = ["mpiexec", "-n", str(t), exe_path, str(N), f"{sp:.6f}", str(k)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    parsed = parse_output(output)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "output": output,
        **parsed
    }
    

def mean_std(values: List[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0
    return mean, std


def run_experiment_means(exe_path: str, N: int, sp: float, k: int, t: int, repeats: int, log_mode: str, run_id: str) -> Dict[str, Any]:
    """
    Runs repeats; keeps only successful parses for averaging.
    If a run crashes (rc!=0) or missing metrics -> counted as failed.
    """
    key_fields = [
        "csr_build_serial_s","csr_build_parallel_s",
        "dense_mult_serial_s","dense_mult_parallel_s",
        "csr_mult_serial_s","csr_mult_parallel_s",
        "dense_csr_speedup_p","dense_csr_speedup_s",
        "dense_speedup_s","csr_speedup_s",
    ]
    

    ok = {f: [] for f in key_fields}
    rc_list = []

    for rep in range(1, repeats + 1):
        r = run_once(exe_path, N, sp, k, t)
        rc = int(r["returncode"])
        rc_list.append(rc)

        missing = [f for f in key_fields if r.get(f) is None]
        is_fail = (rc != 0) or (len(missing) > 0)

        if is_fail:
            continue

        for f in key_fields:
            ok[f].append(float(r[f]))

    n_ok = len(ok[key_fields[0]])
    n_fail = repeats - n_ok

    row: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "matrix_size": N,
        "sparsity": sp,
        "num_mults": k,
        "processes": t,
        "repeats": repeats,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "rc_last": rc_list[-1] if rc_list else 0,
    }

    # If nothing succeeded, keep NaNs
    if n_ok == 0:
        for f in key_fields:
            row[f"{f}_mean"] = np.nan
            row[f"{f}_std"]  = np.nan
        return row

    # Means/stds
    for f in key_fields:
        m, s = mean_std(ok[f])
        row[f"{f}_mean"] = m
        row[f"{f}_std"]  = s

    return row


def append_row_csv(path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--log-mode", choices=["none","fail","all"], default="none",
                    help="Save logs: none (default), fail, or all.")
    ap.add_argument("--skip-experiments", action="store_true",
                    help="Skip running and only plot from existing means CSV.")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(EXE_PATH):
        raise SystemExit(f"[ERROR] Executable not found: {EXE_PATH}")

    if args.skip_experiments:
        if not os.path.exists(MEANS_CSV_PATH):
            raise SystemExit(f"[ERROR] Missing means CSV: {MEANS_CSV_PATH}")
    else:
        # Overwrite means CSV each new full run
        if os.path.exists(MEANS_CSV_PATH):
            os.remove(MEANS_CSV_PATH)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        total = len(MATRIX_SIZE) * len(SPARSITY) * len(NUM_MULTS) * len(PROCESSES)
        c = 0

        # fixed header
        fieldnames = [
            "timestamp","matrix_size","sparsity","num_mults","processes","repeats","n_ok","n_fail","rc_last",
        ]
        # add mean/std columns
        fieldnames += [f"{k}_mean" for k in key_fields]
        fieldnames += [f"{k}_std"  for k in key_fields]
        
        for N in MATRIX_SIZE:
            for sp in SPARSITY:
                for k in NUM_MULTS:
                    
                    for t in PROCESSES:
                        c += 1
                        print(f"[INFO] {c}/{total} | N={N} sp={sp:.2f} k={k} t={t}")

                        row = run_experiment_means(EXE_PATH, N, sp, k, t, args.repeats, args.log_mode, run_id)

                        # If it crashed at least once, note it
                        if row["n_ok"] == 0 or row["n_fail"] > 0 or row["rc_last"] != 0:
                            print(f"[WARN] Partial/failed runs for N={N} sp={sp:.2f} k={k} t={t} "f"(n_ok={row['n_ok']}, n_fail={row['n_fail']}, rc_last={row['rc_last']})")

                        
                        
                        append_row_csv(MEANS_CSV_PATH, row, fieldnames)

        print(f"[INFO] Saved means CSV: {MEANS_CSV_PATH}")
    '''
    if not args.no_plots:
        make_plots(MEANS_CSV_PATH)
    '''

if __name__ == "__main__":
    main()
