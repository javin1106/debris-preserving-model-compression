#!/usr/bin/env bash
# Phase 1a (env-orig, laptop): official evaluation.py on checkpoint 1 (cross-check), then our evaluate.py on all
# 5 released checkpoints, val + test, single-pass + TTA. Usage (WSL, repo root): bash scripts/p1_eval_teacher_local.sh [DATA]
set -eo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${1:-$HOME/mados_data/MADOS}"
OUT="$REPO/runs/p1-official-eval-ckpt1"
mkdir -p "$OUT"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mados
cd "$REPO/third_party/mados"

# Official script, untouched, README usage. Its log goes to third_party/mados/logs/; we move it out afterwards.
for TTA in True False; do
    python marinext/evaluation.py --path "$DATA" --model_path "$REPO/checkpoints/teacher/1" \
        --split test --test_time_augmentations "$TTA" > "$OUT/official_test_tta${TTA}.txt" 2>&1
done
mv logs/evaluating_marinext.log "$OUT/evaluating_marinext.log"

cd "$REPO"
CK=""; TAGS=""
for k in 1 2 3 4 5; do CK="$CK checkpoints/teacher/$k/marinext_$k.pth"; TAGS="$TAGS ckpt$k-envorig"; done
python -m src.debriskd.evaluate --path "$DATA" --ckpt $CK --tag $TAGS \
    --splits val test --modes single tta --phase p1 --setting A
echo "P1_EVAL_DONE"
