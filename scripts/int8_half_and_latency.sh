#!/usr/bin/env bash
# 1) Tuned INT8 (pctends) for the half-width full method (F2 x0.5, seed 0), val + test.
# 2) CPU latency (ONNX Runtime, FP32 vs INT8, 8 and 1 threads) for teacher ckpt5, F2 x0.5, F2 x0.25 s0.
#    Run only on an idle laptop (no training): latency does not depend on the weights, one model per size.
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh; conda activate mados
echo "=== INT8 half width  $(date)"
python -m src.debriskd.quantize --variant pctends --calib_method percentile --keep_ends_fp32 \
    --student runs/p3-F2-w0.5-s0-20261028-2355 2>&1 | grep --line-buffered -E "p4-H|wrote|Error|Traceback"
half=$(ls -d runs/p4-H-F2-w0.5-s0-qpctends-* | tail -1)
echo "=== CPU latency  $(date)"
python -m src.debriskd.quantize --latency runs/p4-H-A-ckpt5-qpctends-20261028-2106 "$half" \
    runs/p4-H-F2-w0.25-s0-qpctends-20261028-2123 2>&1 | grep --line-buffered -E "ms|Error|Traceback"
echo "=== INT8 HALF + LATENCY DONE  $(date). You can close this window."
exec bash
