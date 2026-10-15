#!/usr/bin/env bash
# Run several training jobs in parallel on the laptop GPU (env-orig, WSL), each resumable.
# Usage (inside WSL, repo root):  bash scripts/run_local_jobs.sh JOBFILE
# JOBFILE: one job per line = "<run_id> <extra train.py args...>"; lines starting with # are skipped.
# Re-running the same JOBFILE resumes unfinished runs (train.py reloads runs/<run_id>/resume.pth)
# and skips finished ones (metrics.json present).
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
JOBFILE="$1"
CACHE="${CACHE:-$HOME/mados_data/cache}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source ~/miniforge3/etc/profile.d/conda.sh
set +u; conda activate mados; set -u
cd "$REPO"
pids=()
while read -r rid args; do
    [ -z "$rid" ] || [[ "$rid" == \#* ]] && continue
    if [ -f "runs/$rid/metrics.json" ]; then echo "skip finished $rid"; continue; fi
    mkdir -p "runs/$rid"
    echo "start $rid ($args)"
    python -m src.debriskd.train --cache "$CACHE" --run_id "$rid" $args > "runs/$rid/stdout.log" 2>&1 &
    pids+=($!)
    sleep 20   # stagger start-up
done < "$JOBFILE"
for p in "${pids[@]}"; do wait "$p"; done
echo "ALL_JOBS_DONE $(date)"
