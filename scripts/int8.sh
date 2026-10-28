#!/usr/bin/env bash
# INT8 (setting H): released teachers 1-5 + quarter-width full method (F2) seeds 0-2. CPU only, low priority.
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh; conda activate mados
echo "=== INT8 quantization + accuracy  $(date)"
nice -n 10 python -m src.debriskd.quantize \
    --teacher checkpoints/teacher/{1,2,3,4,5}/marinext_*.pth \
    --student runs/p3-F2-w0.25-s0-20261021-0312 runs/p3-F2-w0.25-s1-20261028-0421 runs/p3-F2-w0.25-s2-20261028-0421 \
    2>&1 | grep --line-buffered -E "p4-H|wrote|Error|Traceback"
echo "=== INT8 DONE  $(date). You can close this window."
exec bash
