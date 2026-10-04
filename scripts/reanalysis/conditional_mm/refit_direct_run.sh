#!/usr/bin/env bash
set -euo pipefail
export JAX_PLATFORM_NAME=cpu XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python -u /data/users/akolchina/epistasis_proteingym/scripts/reanalysis/conditional_mm/refit_direct.py --dataset "$1"
