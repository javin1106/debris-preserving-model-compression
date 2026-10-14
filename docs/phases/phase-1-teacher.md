# Phase 1 — Teacher reproduction (local) + port to the H100

## Goal
Prove the teacher is trustworthy in two environments before anything is built on it.
This phase has two gates. Do not continue past a failed gate — stop and report.

## 1a. Reproduce in `env-orig` (laptop)

### Tasks
- Write `src/debriskd/evaluate.py`: a thin wrapper around the official evaluation logic that
  - loads a checkpoint, runs the official test loader,
  - computes per-class precision, recall, F1, IoU, plus mIoU and macro-F1 **on labelled pixels only**,
  - supports `--tta` (uses the official `utils/test_time_aug.py`) and reports single-pass and TTA separately,
  - writes `runs/<run_id>/metrics.json` and `env.json`.
- Unit test (`tests/test_metrics.py`): a tiny synthetic label map with ignore pixels; check that ignore pixels
  never count as TP/FP/FN and that F1 = 2·IoU/(1+IoU) holds per class.
- Cross-check: for one checkpoint, your numbers must match the official `evaluation.py` output exactly.
- Evaluate all 5 checkpoints on val and test, single-pass and TTA.
- `scripts/summarise_teacher.py` → `results/teacher_repro.csv` and `results/teacher_repro.md`:
  per-class mean ± sd across the 5 checkpoints; debris F1 / recall / IoU next to the paper's 51.3 / 70.4 / 34.6.
- Determine from the paper text/README which setting the paper's Table 5 reports (single model or mean of 5;
  TTA or not). State the answer and its source, or state that it is unknown.

### GATE 1
Pass if the paper's debris F1, recall and IoU each lie within our mean ± 2 sd across the 5 checkpoints
(for the matching TTA setting), or within 1.5 points of our mean.
If it fails: stop. Report the gap, what you checked, and your best hypothesis. Do not proceed.

## 1b. Port to `env-h100`

### Tasks
- Write `docs/ENVIRONMENT.md` section "env-h100" with exact commands. Target:
  - PyTorch 2.x with a CUDA 12 build (the H100 needs CUDA ≥ 11.8)
  - `mmcv` 1.7.x — the lite package, **not** `mmcv-full` (no compiled ops needed by MariNeXt)
  - timm, GDAL and other requirements at the closest compatible versions
- Known patch: `third_party/mados/marinext/mmseg/models/losses/focal_loss.py` imports from `mmcv.ops`
  at top level. Wrap that import in `try/except ImportError` (focal loss is not used by MariNeXt).
  Before patching, grep the whole vendored `mmseg` for other top-level `mmcv.ops` imports reached
  on the import path of the MariNeXt wrapper, and handle them the same way.
  Record every patch (file, diff, reason) in `docs/PATCHES.md`. Store the patches as a `.patch` file in
  `scripts/patches/` and apply them with a script, so the submodule stays clean in git.
- If mmcv 1.7.x will not import with the chosen PyTorch 2.x, try an earlier 2.x release before anything else;
  record what was tried.
- Write the H100 launch helper in `scripts/` matching the access method recorded in `docs/STATUS.md`
  (tmux/nohup script, or a job-script template for a scheduler). Logs and checkpoints to `runs/`.
- Re-run evaluation of all 5 checkpoints in `env-h100` (TF32 disabled, FP32).
- `results/env_parity.md`: per-class differences env-orig vs env-h100.

### GATE 2
Pass if, for every checkpoint, debris F1/recall/IoU differ by ≤ 0.5 points between environments,
and no class differs by more than 1.0 point. Otherwise stop and report.

## 1c. Retrain the teacher on the H100 (only after both gates pass)
- Using the official `train.py` and **all official defaults**, train 3 teachers with seeds 0, 1, 2 on the H100
  (run them in parallel, not with a larger batch).
- Compare their val/test metrics with the 5 released checkpoints → append to `results/teacher_repro.md`.
- **Choose the teacher for later phases** by best **validation** debris F1 among the released checkpoints
  and our retrained ones. Record the choice and why in `docs/STATUS.md` → Decisions. Never choose by test score.

## Deliverables
- `src/debriskd/evaluate.py`, `tests/test_metrics.py`, `scripts/summarise_teacher.py`
- `results/teacher_repro.{csv,md}`, `results/env_parity.md`
- `docs/ENVIRONMENT.md` (env-h100 section), `docs/PATCHES.md`, `scripts/patches/*.patch`, H100 launch script
- Chosen teacher checkpoint path recorded in `docs/STATUS.md`

## Acceptance criteria
- [ ] GATE 1 passed and documented
- [ ] GATE 2 passed and documented
- [ ] 3 retrained teachers evaluated; teacher chosen on validation
- [ ] `pytest -q` passes
