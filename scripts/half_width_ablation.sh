#!/usr/bin/env bash
# Half width (x0.5) ablation, 3 seeds: D2, E2, C-long (9 runs, 2 at a time, auto-resume on crash), then summaries.
cd "$(dirname "$0")/.."
export SHOW_LOGS=1 SLOTS=2 RETRIES=3
echo "=== Half width ablation: D2, E2, C-long x 3 seeds (160 rounds)  $(date)"
bash scripts/run_queue.sh configs/jobs/half_width_ablation.txt
source ~/miniforge3/etc/profile.d/conda.sh; conda activate mados
python scripts/summarise_seeds.py --width 0.5
echo "=== HALF-WIDTH ABLATION DONE  $(date). You can close this window."
exec bash
