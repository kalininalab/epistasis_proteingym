#!/usr/bin/env bash
set -euo pipefail
root=/data/users/akolchina/epistasis_proteingym
base="$root/results/reanalysis/tsuboyama/vespa_family"
export HF_HOME=/data/users/akolchina/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mode="$1"; dataset="$2"; limit="${3:-none}"; namespace="${4:-full}"
[[ "$namespace" == full ]] || base="$base/$namespace"
if [[ "$mode" == score ]]; then
  extra=(); [[ "$limit" == none ]] || extra=(--max-doubles "$limit")
  exec /data/users/akolchina/envs/vespa_family/bin/python -u "$root/scripts/reanalysis/vespa_family/score_prott5.py" \
    --checkpoint Rostlab/prot_t5_xl_uniref50 --revision 973be27c52ee6474de9c945952a8008aeb2a1a73 \
    --model-key ProtT5 --default-column ProtT5 --dataset "$dataset" --output "$base/scores/ProtT5/$dataset" \
    --batch-size 4 --pll-sequence-chunk 32 "${extra[@]}"
else
  export JAX_PLATFORM_NAME=cpu XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
  exec /nethome/akolchina/miniconda3/envs/protease-pipeline/bin/python -u "$root/scripts/reanalysis/vespa_family/fit.py" \
    --scores "$base/scores/ProtT5/$dataset/scores.csv" --output "$base/fits/ProtT5/$dataset"
fi
