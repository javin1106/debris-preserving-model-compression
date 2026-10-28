#!/usr/bin/env bash
# Run a job file as a queue on the laptop GPU (env-orig, WSL): SLOTS jobs at a time, the next job
# starts as soon as a slot frees. A job that exits without metrics.json (e.g. after a GPU driver reset,
# "CUDA error: unknown error") is resumed from runs/<run_id>/resume.pth after a pause, up to RETRIES times.
# Usage (inside WSL, repo root):  [SLOTS=2] [RETRIES=3] [SHOW_LOGS=1] bash scripts/run_queue.sh JOBFILE
# JOBFILE format as for run_local_jobs.sh: "<run_id> <extra train.py args...>", # lines skipped.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
JOBFILE="$1"
SLOTS="${SLOTS:-2}"
export RETRIES="${RETRIES:-3}" CACHE="${CACHE:-$HOME/mados_data/cache}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source ~/miniforge3/etc/profile.d/conda.sh
set +u; conda activate mados; set -u
cd "$REPO"

run_one() {
    local rid="$1"; shift
    local try
    for try in $(seq 0 "$RETRIES"); do
        if [ -f "runs/$rid/metrics.json" ]; then echo "finished $rid $(date)"; return 0; fi
        if [ "$try" -gt 0 ]; then echo "retry $try/$RETRIES for $rid in 120 s (resumes from last saved round)"; sleep 120; fi
        mkdir -p "runs/$rid"
        echo "start $rid ($*) $(date)"
        python -m src.debriskd.train --cache "$CACHE" --run_id "$rid" "$@" >> "runs/$rid/stdout.log" 2>&1
    done
    [ -f "runs/$rid/metrics.json" ] && echo "finished $rid $(date)" || echo "FAILED $rid after $RETRIES retries"
}
export -f run_one

jobs_list="$(grep -vE '^[[:space:]]*(#|$)' "$JOBFILE")"
if [ "${SHOW_LOGS:-0}" = "1" ]; then
    # Live view: print each finished epoch (and final results) of every job in the file.
    # Polling, because inotify misses writes to files under /mnt/c in WSL.
    logs=(); while read -r rid _; do logs+=("runs/$rid/train.log"); done <<< "$jobs_list"
    tail -n 0 --use-polling -s 5 -F "${logs[@]}" 2>/dev/null | grep --line-buffered -E "==>|epoch [0-9]+ \||FINAL|DONE|Error" &
    tailpid=$!
fi
echo "$jobs_list" | xargs -P "$SLOTS" -L 1 bash -c 'run_one "$@"' _
[ -n "${tailpid:-}" ] && kill "$tailpid" 2>/dev/null
echo "ALL_JOBS_DONE $(date)"
