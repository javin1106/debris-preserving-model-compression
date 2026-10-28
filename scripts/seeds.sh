#!/usr/bin/env bash
# Seeds 1-2 (2026-10-28) for normal training, debris x2, full method at x0.25, 160 rounds. Live progress shown.
cd "$(dirname "$0")/.."
export SHOW_LOGS=1
echo "=== Wave 1: normal (seed 1) + debris x2 (seed 1)  $(date)"; bash scripts/run_local_jobs.sh configs/jobs/seeds_wave1.txt
echo "=== Wave 2: normal (seed 2) + debris x2 (seed 2)  $(date)"; bash scripts/run_local_jobs.sh configs/jobs/seeds_wave2.txt
echo "=== Wave 3: full method (seeds 1 and 2)  $(date)";      bash scripts/run_local_jobs.sh configs/jobs/seeds_wave3.txt
echo "=== ALL SEED RUNS DONE  $(date). You can close this window."
exec bash
