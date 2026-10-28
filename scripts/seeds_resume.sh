#!/usr/bin/env bash
# Resume the seed 1-2 repeats (2026-10-28) after GPU driver crashes: 2-slot queue, auto-resume on crash. Live progress shown.
cd "$(dirname "$0")/.."
export SHOW_LOGS=1 SLOTS=2 RETRIES=3
echo "=== Seed repeats (resume): full method s1/s2, debris x2 s1/s2, normal s2  $(date)"
bash scripts/run_queue.sh configs/jobs/seeds_resume.txt
echo "=== ALL SEED RUNS DONE  $(date). You can close this window."
exec bash
