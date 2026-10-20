#!/usr/bin/env bash
# Phase 3.4 smoke tests (laptop, env-orig): each setting B-F at width 1/4, 50 patches, 2 epochs.
# Plus an equivalence check: D with lam_L = lam_U = 0 (teacher runs, contributes nothing) must give exactly B's losses.
# Usage (WSL, repo root): bash scripts/smoke_settings.sh
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source ~/miniforge3/etc/profile.d/conda.sh
set +u; conda activate mados; set -u
cd "$REPO"
TS=$(date -u +%Y%m%d-%H%M)
COMMON="--cache $HOME/mados_data/cache --width 0.25 --subset 50 --epochs 2 --final_eval false --phase smoke"
for S in B C D E F; do
    python -m src.debriskd.train $COMMON --config configs/settings/$S.yaml --run_id smoke-$S-$TS > /dev/null 2>&1
    echo "== $S: $(grep -cE '^.*epoch [0-9]+ \|' runs/smoke-$S-$TS/train.log) epochs, exit-ok: $(grep -c DONE runs/smoke-$S-$TS/train.log)"
    python - "runs/smoke-$S-$TS/epochs.jsonl" <<'EOF'
import json, math, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    kd = r.get("kd", {})
    finite = all(math.isfinite(v) for v in [r["train_loss"]] + [v for k, v in kd.items() if isinstance(v, float)])
    print(f"   epoch {r['epoch']}: train loss {r['train_loss']:.6f}"
          + (f" | sup {kd['sup']:.4f} kd_lab {kd['kd_lab']:.4f} kd_ign {kd['kd_ign']:.4f} lam_U {kd['lam_U']:.2f}"
             f" | ignored px used {100*kd['frac_ign_used']:.1f}% | teacher classes there: {kd['teacher_hist_ign']}" if kd else "")
          + f" | finite {finite}")
EOF
done
python -m src.debriskd.train $COMMON --config configs/settings/D.yaml --lam_L 0 --lam_U 0 --run_id smoke-Dzero-$TS > /dev/null 2>&1
python - "runs/smoke-B-$TS/epochs.jsonl" "runs/smoke-Dzero-$TS/epochs.jsonl" <<'EOF'
import json, sys
b = [json.loads(l)["train_loss"] for l in open(sys.argv[1])]
d = [json.loads(l)["train_loss"] for l in open(sys.argv[2])]
print("== equivalence B vs D(lambda=0): B", b, "| D0", d, "| identical:", b == d)
EOF
grep -h "CE class weights" runs/smoke-B-$TS/train.log runs/smoke-C-$TS/train.log | sed 's/.*CE class weights/CE weights/'
