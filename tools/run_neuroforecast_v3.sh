#!/usr/bin/env bash
# Full NeuroForecast v3 study (protocol: experiments/neuroforecast_v3_protocol.json).
# PY: python of the v3 environment (packages: E:/kaggle/AI4S/experiments/neuroforecast_v3/venv_freeze.txt).
set -euo pipefail
PY=${PY:-python}
OUT=${OUT:-E:/kaggle/AI4S/experiments/neuroforecast_v3}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$(dirname "$0")/.."
mkdir -p "$OUT/logs"
run() { "$PY" -u -X utf8 -B -W ignore tools/neuroforecast_v3_run.py "$@" --out "$OUT"; }
run lock > "$OUT/logs/lock.log" 2>&1
run tabular --jobs 16 > "$OUT/logs/tabular.log" 2>&1 &
TABULAR=$!
run pct --only pct > "$OUT/logs/pct.log" 2>&1
run tabpfn > "$OUT/logs/tabpfn.log" 2>&1
run pct --only pct_no_context pct_no_relations pct_no_aux > "$OUT/logs/pct_ablations.log" 2>&1
wait "$TABULAR"
"$PY" -u -X utf8 -B -W ignore tools/neuroforecast_v3_analyze.py --out "$OUT" > "$OUT/logs/analyze.log" 2>&1
echo done
