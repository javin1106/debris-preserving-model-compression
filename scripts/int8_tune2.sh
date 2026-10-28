#!/usr/bin/env bash
# INT8 setting search, step 2 (VALIDATION only, teacher ckpt 1): percentile calibration + first/last conv FP32 together.
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh; conda activate mados
echo "=== INT8 setting search 2 (val, teacher ckpt 1)  $(date)"
nice -n 10 python -m src.debriskd.quantize --teacher checkpoints/teacher/1/marinext_1.pth --splits val \
    --variant pctends --calib_method percentile --keep_ends_fp32 2>&1 | grep --line-buffered -E "p4-H|wrote|Error|Traceback"
echo "=== INT8 SEARCH 2 DONE  $(date). You can close this window."
exec bash
