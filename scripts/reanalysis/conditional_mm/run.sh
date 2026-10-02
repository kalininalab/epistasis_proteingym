#!/usr/bin/env bash
set -euo pipefail
root=/data/users/akolchina/epistasis_proteingym
case "$1" in
 score) exec /nethome/akolchina/miniconda3/envs/ofs/bin/python -u "$root/scripts/reanalysis/conditional_mm/score_all.py" --shard "$2" --n-shards "$3";;
 fit) export JAX_PLATFORM_NAME=cpu XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; exec /nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python -u "$root/scripts/reanalysis/conditional_mm/fit.py" --dataset "$2";;
 *) exit 2;;
esac
