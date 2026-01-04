#!/bin/bash

# Usage: ./run_mpi.sh <executable> <num_processes>

if [ $# -lt 2 ]; then
    echo "Usage: $0 <executable> <num_processes>"
    exit 1
fi

EXE=$1                      # executable path
NP=$2                       # number of processes
MACHINEFILE=~/machinefile   # machinefile path

PPN=4 # default number of processes per node

# specify custom number of ppn at the optional 3rd argument
if [ $# -gt 2 ]; then
    PPN=$3
fi

mpiexec -np $NP -f $MACHINEFILE -ppn $PPN $EXE