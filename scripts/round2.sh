#!/usr/bin/env bash
# Round 2 (2026-10-21): class-weighted KD at x0.25, 160 epochs. Two waves, two jobs each, live progress shown.
cd "$(dirname "$0")/.."
export SHOW_LOGS=1
echo "=== Wave 1: E2 (teacher, unlabelled px) + F2 (full method)  $(date)"
bash scripts/run_local_jobs.sh configs/jobs/round2_wave1.txt
echo "=== Wave 2: D2 (teacher, labelled px) + C-long (extra debris weight)  $(date)"
bash scripts/run_local_jobs.sh configs/jobs/round2_wave2.txt
echo "=== ALL ROUND-2 RUNS DONE  $(date). You can close this window."
exec bash
