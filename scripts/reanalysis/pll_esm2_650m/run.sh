#!/usr/bin/env bash
set -euo pipefail
root=/data/users/akolchina/epistasis_proteingym
exec /nethome/akolchina/miniconda3/envs/ofs/bin/python -u "$root/scripts/reanalysis/pll_esm2_650m/score.py" "$@"
