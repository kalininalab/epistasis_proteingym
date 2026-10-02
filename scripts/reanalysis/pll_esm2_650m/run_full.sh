#!/usr/bin/env bash
set -euo pipefail
root=/data/users/akolchina/epistasis_proteingym
case "$1" in
  score)
    dataset="$2"
    exec /nethome/akolchina/miniconda3/envs/ofs/bin/python -u "$root/scripts/reanalysis/pll_esm2_650m/score.py" \
      --dataset "$dataset" --output "$root/results/reanalysis/tsuboyama/pll_esm2_650m/scores/$dataset" --batch-size 32
    ;;
  fit)
    dataset="$2"
    export JAX_PLATFORM_NAME=cpu
    export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec /nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python -u "$root/scripts/reanalysis/pll_esm2_650m/fit.py" \
      --scores "$root/results/reanalysis/tsuboyama/pll_esm2_650m/scores/$dataset/scores.csv" \
      --output "$root/results/reanalysis/tsuboyama/pll_esm2_650m/fits/$dataset"
    ;;
  *) echo 'expected score DATASET or fit DATASET' >&2; exit 2;;
esac
