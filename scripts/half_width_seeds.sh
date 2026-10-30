#!/usr/bin/env bash
# Overnight: half width (x0.5) seeds 1-2 for normal training + full method (2 at a time, auto-resume on crash),
# then tuned INT8 (pctends, val + test) for the two new full-method models, then regenerate the summaries.
cd "$(dirname "$0")/.."
export SHOW_LOGS=1 SLOTS=2 RETRIES=3
echo "=== Half width seeds 1-2 (160 rounds)  $(date)"
bash scripts/run_queue.sh configs/jobs/half_width_seeds.txt
source ~/miniforge3/etc/profile.d/conda.sh; conda activate mados
for s in 1 2; do
    run=$(ls -d runs/p3-F2-w0.5-s$s-* 2>/dev/null | tail -1)
    if [ -n "$run" ] && grep -q DONE "$run/train.log"; then
        echo "=== INT8 $run  $(date)"
        python -m src.debriskd.quantize --variant pctends --calib_method percentile --keep_ends_fp32 \
            --student "$run" 2>&1 | grep --line-buffered -E "p4-H|wrote|Error|Traceback"
    else
        echo "=== skipping INT8 for seed $s (training not finished)"
    fi
done
python scripts/summarise_int8.py; python scripts/summarise_headline.py
echo "=== OVERNIGHT RUNS DONE  $(date). You can close this window."
exec bash
