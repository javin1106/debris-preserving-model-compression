# Phase 5 — Analysis, figures, report

> Hardware note (2026-10-14): "3050" below means the laptop's RTX 4060 Laptop GPU (8 GB).
> See `docs/STATUS.md` → Corrections.

## Goal
Turn all runs into traceable tables, figures and a written report that answers the research questions honestly.

## Research questions to answer
1. How do debris F1, recall and IoU change as the student shrinks? Where is the cliff?
2. Do rare classes (debris) degrade earlier than common ones (water)?
3. Does extra class weighting (C) solve it, or does distillation add something?
4. Is teacher guidance on ignored pixels (E) more useful than on labelled pixels only (D)?
5. Which compression route — narrower student, pruning, INT8 — gives the best debris accuracy per unit of size, speed and memory?

## Tasks

### 5.1 Aggregation — `scripts/aggregate.py`
- Walk `runs/`, read every `config.yaml`, `metrics.json`, `env.json` → `results/master.csv` (one row per run).
- Fail loudly if a run is missing files. Never fill gaps by hand.

### 5.2 Tables (generated)
- Main table: setting × width, mean ± sd over seeds, test split: debris F1/recall/IoU, mIoU, macro-F1, params, latency (3050), size.
- Per-class degradation: F1 change from teacher to each compressed model, all 15 classes.
- Efficiency table from Phase 4.

### 5.3 Figures (matplotlib; PNG + PDF; readable at A4)
- Debris F1 vs parameters (log x), all routes on one plot, error bars = sd.
- Debris F1 vs latency on RTX 3050 and on CPU.
- Per-class degradation heatmap (classes × models).
- Qualitative grid: RGB composite of the patch, ground truth, teacher, B, F, for 6 test patches.
  Choose patches by a fixed rule written in the script (e.g. the 6 test patches with the most labelled
  debris pixels) — no hand-picking. Show ignored pixels in a neutral colour.

### 5.4 Report — `results/REPORT.md`
Sections: aim; data and setup (exact recipe, seeds, hardware, environments, patches); methods (loss with equations);
results (tables and figures, every number referenced to run IDs in `master.csv`); answers to the 5 research questions;
limitations (teacher error propagation, dataset scope, no real FPGA measurement, CPU/3050 as proxies);
recommended deployable model and why.

- Negative or null results are stated plainly.
- Distinguish clearly between the paper's published numbers and ours.

## Deliverables
- `scripts/aggregate.py`, `results/master.csv`, all tables and figures, `results/REPORT.md`

## Acceptance criteria
- [ ] Every number in `REPORT.md` traceable to a run ID
- [ ] All five research questions answered, including uncertainty (sd, seed spread)
- [ ] Figures render cleanly and are legible when printed at A4
