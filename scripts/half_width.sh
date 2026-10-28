#!/usr/bin/env bash
# Half width (x0.5), seed 0, 160 rounds: normal training + full method, 2 at a time, auto-resume on crash. Live progress shown.
cd "$(dirname "$0")/.."
export SHOW_LOGS=1 SLOTS=2 RETRIES=3
echo "=== Half width: normal training + full method (seed 0, 160 rounds)  $(date)"
bash scripts/run_queue.sh configs/jobs/half_width.txt
echo "=== HALF-WIDTH RUNS DONE  $(date). You can close this window."
exec bash
