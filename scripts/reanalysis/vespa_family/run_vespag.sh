#!/usr/bin/env bash
set -euo pipefail
root=/data/users/akolchina/epistasis_proteingym
base="$root/results/reanalysis/tsuboyama/vespa_family"
export HF_HOME=/data/users/akolchina/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/data/users/akolchina/software/VespaG:${PYTHONPATH:-}
mode="$1"; dataset="$2"; limit="${3:-none}"; namespace="${4:-vespag_full}"
[[ "$namespace" == vespag_full ]] || base="$base/$namespace"
if [[ "$mode" == score ]]; then
  extra=(); [[ "$limit" == none ]] || extra=(--max-doubles "$limit")
  exec /data/users/akolchina/envs/vespa_family/bin/python -u "$root/scripts/reanalysis/vespa_family/score_vespag.py" \
    --dataset "$dataset" --output "$base/scores/VespaG/$dataset" --batch-size 2 "${extra[@]}"
else
  export JAX_PLATFORM_NAME=cpu XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
  exec /nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python -u "$root/scripts/reanalysis/vespa_family/fit_vespag.py" \
    --scores "$base/scores/VespaG/$dataset/scores.csv" --output "$base/fits/VespaG/$dataset"
fi
