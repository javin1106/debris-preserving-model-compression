# Phase 2 — Student builder + width sweep (setting B)

## Goal
Build width-scaled MariNeXt students, train each with the official recipe on labels only,
and find the width at which debris accuracy collapses (the "cliff").

## Decision to confirm with the user first
The official recipe already includes class weighting (`--weight_param`). Proposed definitions:
- **B** = official recipe exactly as is (keeps parity with the paper).
- **C** = B + an *extra* debris-focused reweighting of the supervised loss.
Ask the user to confirm (or redefine) B and C. Record in `docs/STATUS.md` → Decisions.

## Tasks

### 2.1 `src/debriskd/students.py`
- `build_student(width: float) -> nn.Module` that reuses the official MariNeXt classes and config,
  scaling: encoder embed dims (32, 64, 160, 256) × width (round to int, minimum 4),
  decoder `channels` and `ham_channels` 256 × width.
  Keep depths, MLP ratios, in_chans = 11, num_classes = 15, and everything else unchanged.
- Check every divisibility constraint in the encoder and decoder (e.g. GroupNorm group count in the decoder
  must divide the channel count; use `min(32, channels)` groups if needed). Document any change in a docstring
  and in `docs/STATUS.md`.
- `width = 1.0` must reproduce the teacher architecture exactly (same parameter count).

### 2.2 Tests — `tests/test_students.py`
- Forward pass on a random 1×11×240×240 tensor for widths 1, ½, ¼, ⅛: output shape matches the teacher's.
- Parameter counts within 2% of: 1.0 → 4,238,063; ½ → ≈1,130,751; ¼ → ≈318,311; ⅛ → ≈97,395.
  If the measured teacher count from Phase 0 differs from 4,238,063, use the measured one and note it.

### 2.3 `src/debriskd/train.py`
- Copy the official `train.py` into our package (keep a header comment pointing to the original + commit hash).
- Add `--width` (default 1.0) to build a student instead of the teacher. **Change nothing else in the recipe.**
- Add run-ID / `runs/` logging per `CLAUDE.md`, per-epoch checkpointing and resume.
- Keep model selection exactly as the official script does it (e.g. best validation epoch), and log which epoch was selected.
- Smoke test locally: `--subset 50 --epochs 1` for width ¼ in `env-orig`.

### 2.4 Width sweep on the H100
- Train setting B at widths ½, ¼, ⅛, seed 0 (3 runs, in parallel).
- Evaluate each on val and test with `evaluate.py` (single-pass; TTA reported separately).
- Optional if the cliff is unclear: add widths ⅜ and 3⁄16 (ask the user first).

### 2.5 Summary — `scripts/summarise_sweep.py`
- `results/width_sweep.csv`: width, params, debris F1/recall/IoU, mIoU, macro-F1 (val and test).
- `results/fig_width_sweep.png`: debris F1 vs parameters (log x-axis), teacher shown as a reference point;
  plus per-class F1 vs width to show which classes drop first.
- Short written note: where the cliff appears **on validation**, and which classes collapse first.

## GATE (user decision)
Propose 1–2 widths for Phase 3: typically the smallest width before the validation cliff and the first width
after it. Explain the choice using **validation** numbers only. Wait for the user to confirm.

## Deliverables
- `src/debriskd/students.py`, `src/debriskd/train.py`, `tests/test_students.py`
- `results/width_sweep.csv`, `results/fig_width_sweep.png`, sweep note
- Decisions on B/C definition and chosen widths in `docs/STATUS.md`

## Acceptance criteria
- [ ] Student builder tested; width 1.0 reproduces the teacher exactly
- [ ] 3 sweep runs completed with logged configs, seeds, env
- [ ] Sweep table and figure generated from `runs/` by script
- [ ] Widths for Phase 3 proposed from validation and confirmed by the user
