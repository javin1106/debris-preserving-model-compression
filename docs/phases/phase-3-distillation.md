# Phase 3 — Distillation experiments (settings C–F)

## Goal
Implement the distillation loss, tune it on validation, and compare settings B–F at the widths
chosen in Phase 2, with 3 seeds each.

## Tasks

### 3.1 `src/debriskd/kd_loss.py`
Implement exactly the loss in `docs/PROJECT_BRIEF.md`:

```python
# t: teacher logits, s: student logits, y: labels with IGNORE; shapes [B, C, H, W] and [B, H, W]
p_t  = softmax(t / T, dim=1)
kl   = (p_t * (log_softmax(t / T, 1) - log_softmax(s / T, 1))).sum(1) * T**2   # [B, H, W]
conf = softmax(t, dim=1).amax(1)                                               # confidence at T = 1
w    = ((conf - tau) / (1 - tau)).clamp(0, 1)
d    = where(t.argmax(1) == DEBRIS, debris_w, 1.0).clamp(max=debris_w_max)
lab  = (y != IGNORE);  ign = ~lab
L_kd_lab = kl[lab].mean()                         if lab.any() else 0
L_kd_ign = (w * d * kl)[ign & (w > 0)].mean()     if (ign & (w > 0)).any() else 0
L = L_sup + lam_L * L_kd_lab + lam_U(epoch) * L_kd_ign
```
- Use `log_softmax` for numerical stability (no `log` of probabilities).
- `L_sup` is the official supervised loss, unchanged (setting C adds its extra reweighting here).
- `lam_U(epoch)`: linear warm-up from 0 to λ_U over the first `kd_warmup_epochs` (default 5).
- Read DEBRIS and IGNORE from the dataset definitions found in Phase 0 — never hard-code a guess.
- Return the individual terms too, for logging.

### 3.2 Tests — `tests/test_kd_loss.py`
- Identical teacher and student logits → both KD terms are 0.
- Ignored pixels never enter `L_sup` or `L_kd_lab`; labelled pixels never enter `L_kd_ign`.
- Pixels with conf ≤ τ contribute nothing; w is in [0, 1].
- λ_L = λ_U = 0 reproduces `L_sup` exactly.
- Gradient flows to the student only (teacher params have no grad).

### 3.3 Integrate into `train.py`
- Flags: `--teacher_ckpt`, `--kd_T`, `--kd_tau`, `--lam_L`, `--lam_U`, `--debris_w`, `--debris_w_max`,
  `--kd_warmup_epochs`, `--extra_debris_ce_weight` (setting C).
- Teacher: loaded from the checkpoint chosen in Phase 1, `eval()`, `requires_grad_(False)`, forward under `torch.no_grad()`
  on the **same augmented batch** the student sees (run live — no cached logits).
- Log per epoch: each loss term; fraction of ignored pixels with w > 0; class histogram of teacher argmax on those pixels.
- One YAML per setting in `configs/` (B, C, D, E, F) so every run is reproducible from its config.

### 3.4 Smoke tests (laptop, `env-orig`)
For each of C–F: `--subset 50 --epochs 1` at width ¼. Losses finite, logs written.

### 3.5 Tuning on validation (H100, seed 0)
- Grid: T ∈ {1, 2, 4}; τ ∈ {0.6, 0.7, 0.8, 0.9}; λ_L, λ_U ∈ {0.5, 1.0}; debris_w ∈ {2, 4} (F only);
  setting C: extra debris weight ∈ {2, 4}. Reduce the grid if it exceeds ~60 runs; ask the user first.
- Tune at one chosen width; reuse the best config at the other width.
- Selection metric: **validation** debris F1; tie-break with validation macro-F1.
- `results/kd_tuning.csv` with every grid point.

### 3.6 Final runs (H100)
- For each chosen width: settings B, C, D, E, F × seeds {0, 1, 2} with their tuned configs
  (B seed 0 can reuse the Phase 2 run if the code path is identical; otherwise rerun).
- Evaluate on test **once**, after configs are frozen.

### 3.7 Summary — `scripts/summarise_kd.py`
- `results/kd_main.csv` and `results/kd_main.md`: mean ± sd of debris F1/recall/IoU, mIoU, macro-F1 per setting × width.
- Paired comparisons B→C, B→D, D→E, E→F: difference of means with sd; state plainly when a difference is
  within run-to-run noise.
- Diagnostics: fraction of ignored pixels used; what classes the teacher predicts there.

## Deliverables
- `src/debriskd/kd_loss.py`, updated `train.py`, `configs/*.yaml`, `tests/test_kd_loss.py`
- `results/kd_tuning.csv`, `results/kd_main.{csv,md}`

## Acceptance criteria
- [ ] All unit tests pass
- [ ] Hyperparameters chosen on validation only, documented in `docs/STATUS.md` → Decisions
- [ ] 5 settings × chosen widths × 3 seeds completed and summarised
- [ ] Results reported honestly, including any setting that does not help

## Pitfalls
- Don't let the KD terms swamp `L_sup` early: keep the warm-up.
- Don't average KL over all pixels when only a few pass τ — average over eligible pixels, as specified.
- If almost no ignored pixels pass τ, report it; lower τ only via the validation grid.
