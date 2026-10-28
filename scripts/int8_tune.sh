#!/usr/bin/env bash
# INT8 setting search on VALIDATION only (teacher ckpt 1): one change at a time vs plain MinMax/uint8.
# rr = 7-bit weights (AVX2 U8S8 saturation), pct = percentile 99.999 calibration, ends = first/last conv FP32,
# s8 = symmetric int8 activations. CPU only, low priority.
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh; conda activate mados
Q="nice -n 10 python -m src.debriskd.quantize --teacher checkpoints/teacher/1/marinext_1.pth --splits val"
echo "=== INT8 setting search (val, teacher ckpt 1)  $(date)"
for v in "rr --reduce_range" "pct --calib_method percentile" "ends --keep_ends_fp32" "s8 --act_type int8"; do
    set -- $v
    name=$1; shift
    $Q --variant "$name" "$@" 2>&1 | grep --line-buffered -E "p4-H|wrote|Error|Traceback"
done
echo "=== INT8 SEARCH DONE  $(date). You can close this window."
exec bash
