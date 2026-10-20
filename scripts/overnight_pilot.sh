#!/usr/bin/env bash
# Overnight chain (2026-10-21): wait for the 160-epoch x0.25 run to finish, then KD pilot waves 1 and 2.
cd "$(dirname "$0")/.."
echo "Waiting for the 160-round quarter run to finish..."
until [ -f runs/p2-Blong-w0.25-s0-20261020-1829/metrics.json ]; do sleep 60; done
echo "Wave 1: D + E ($(date))"; bash scripts/run_local_jobs.sh configs/jobs/pilot_wave1.txt
echo "Wave 2: F + C ($(date))"; bash scripts/run_local_jobs.sh configs/jobs/pilot_wave2.txt
echo "ALL PILOT RUNS DONE ($(date))"
