# CLAUDE.md — Debris-Preserving Model Compression (MariNeXt → students on MADOS)

College research project (CV / deep learning). Team of 2 + faculty guide.
Full research spec: @docs/PROJECT_BRIEF.md

## How to work in this repo — read first, every session

1. Open `docs/STATUS.md`. It names the **current phase**. Work only on that phase.
2. Read the matching file in `docs/phases/` completely before writing code.
3. Never start the next phase on your own. When the phase's acceptance criteria are met,
   stop, summarise what was done against each criterion, and wait for the user.
4. Update `docs/STATUS.md` at the end of every working session: what was done,
   what is blocked, decisions taken, open questions.
5. If something in the docs contradicts the actual source code of the official repo,
   **the code wins**. Record the discrepancy in `docs/STATUS.md` under "Corrections".

## Non-negotiable research rules

- **Never invent numbers.** Every metric in `results/` must come from a run that
  actually executed, traceable to a run ID. If a run was not executed, say so.
- **Unlabelled pixels are unknown, not water.** They carry the dataset's ignore index.
  They are excluded from the supervised loss and from all evaluation metrics.
  Only the distillation term may use them, and only as soft, confidence-weighted guidance.
- **The test split is touched only for final reporting.** All hyperparameters
  (T, τ, λ_L, λ_U, debris weight, student width choice, pruning ratio) are chosen on validation.
- **Fixed training recipe.** Every model uses the official MariNeXt recipe (epochs, batch size,
  LR schedule, augmentation) unless a phase file explicitly says otherwise.
  Do NOT increase batch size to use the H100 better — run jobs in parallel instead.
- **Speed is measured only on the local RTX 4060 Laptop GPU and CPU.** Never report H100 latency as deployment speed.
- **Same conditions for every compared model:** split, bands, resolution, preprocessing,
  precision, hardware, evaluation code.
- **Seeds:** every run sets and logs its seed. Final settings run with 3 seeds; report mean ± sd.

## Ask the user before you

- install a new major dependency or change a pinned version
- change anything in the official training recipe
- delete or overwrite data, checkpoints or results
- modify any file under `third_party/mados/` (only documented patches are allowed — see below)
- launch anything that will run longer than ~30 minutes

## Repository layout (target — Phase 0 creates it)

```
third_party/mados/        official MariNeXt repo (git submodule, pinned commit). Do not edit
                          except patches recorded in docs/PATCHES.md
src/debriskd/             our code (package)
  students.py             width-scaled MariNeXt students
  kd_loss.py              distillation loss (labelled + confidence-weighted ignored-pixel terms)
  train.py                copy of official train.py, extended with --width and KD options
  evaluate.py             per-class metrics on labelled pixels, single-pass and TTA
  prune.py  quantize.py  benchmark.py  count_params.py
configs/                  one YAML per experiment setting
scripts/                  launch scripts (local + H100)
tests/                    pytest unit tests (shapes, param counts, loss masking)
data/                     MADOS (gitignored)
checkpoints/              teacher + trained models (gitignored)
runs/<run_id>/            per-run logs, config copy, metrics.json (gitignored)
results/                  committed CSV/MD summaries and figures
docs/                     brief, status, phases, environment notes, patches
```

## Two environments

| Name | Where | Stack | Purpose |
|---|---|---|---|
| `env-orig` | local laptop, Windows 11 + WSL2 Ubuntu (RTX 4060 Laptop 8 GB, CPU, RAM — see `docs/ENVIRONMENT.md`) | Python 3.8, PyTorch 1.11 + CUDA 11.3, mmcv-full 1.6.0, timm 0.4.12, GDAL (as in official README) | exact reproduction of the paper; debugging; latency benchmarks |
| `env-h100` | institute H100 (SSH) | PyTorch 2.x + CUDA 12, `mmcv` 1.7.x (lite, no compiled ops), GDAL | all long training runs |

- The H100 (Hopper) is not supported by CUDA 11.3, so the official pinned stack will not run there.
- MariNeXt itself only uses pure-Python mmcv pieces. The one known blocker for mmcv-lite is a
  top-level `from mmcv.ops import ...` in `marinext/mmseg/models/losses/focal_loss.py`
  (patch documented in Phase 1).
- Shared code in `src/debriskd/` must run in **both** environments: no PyTorch-2-only APIs
  (e.g. `torch.compile`) unless guarded by a version check.
- The laptop runs Windows, so `env-orig` lives in WSL2 (Ubuntu); native Windows + mmcv-full + GDAL is not worth the pain.

## Reproducibility conventions

- Run ID format: `<phase>-<setting>-w<width>-s<seed>-<YYYYMMDD-HHMM>`, e.g. `p3-F-w0.25-s1-20261020-1412`.
- Each run writes to `runs/<run_id>/`: `config.yaml` (fully resolved), `train.log`, `metrics.json`
  (per-class precision, recall, F1, IoU on val and/or test; mIoU; macro-F1), `env.json`
  (python/torch/cuda/mmcv versions, GPU name, git commit hashes of this repo and the submodule).
- Disable TF32 explicitly in all training and evaluation for parity across GPUs:
  `torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False`.
- Teacher reproduction is FP32. If mixed precision is ever used for students, use it for **all** students.
- Results tables in `results/` are generated by scripts from `runs/`, never typed by hand.

## Coding conventions

- Python, type hints, docstrings on public functions. Keep files small and single-purpose.
- Every training/eval script accepts `--subset N` (use only N patches) and `--epochs` override,
  so a smoke test finishes in under 2 minutes on the local GPU. Always smoke-test before a full run.
- Write a unit test for anything with masking or indexing logic (ignore index handling,
  confidence weights, per-class metrics). Run `pytest -q` before saying a task is done.
- Long runs on the H100: launch via `scripts/` (tmux/nohup over SSH),
  checkpoint every epoch, support resuming.

## Key facts (verify against source before relying on them)

- Official code: https://github.com/gkakogeorgiou/mados (MIT). Pretrained: 5 MariNeXt checkpoints (link in README).
- Data: MADOS, Zenodo doi:10.5281/zenodo.10664073. Sentinel-2, 15 classes, sparse pixel labels.
- Config `marinext/configs/marinext.tiny.240x240.mados.py`: 11 input bands, 240×240 patches,
  MSCAN-Tiny encoder (embed dims 32-64-160-256, depths 3-3-5-2) + LightHam decoder.
- Analytic parameter count (ours, from the config): teacher 4,238,063. Students at width ×½, ×¼, ×⅛:
  ≈1.13 M, ≈0.32 M, ≈0.10 M. Confirm with `count_params.py`.
- Published MariNeXt debris-class results (paper Table 5): F1 51.3, recall 70.4, IoU 34.6.
  These are the paper's, not ours.
- Official recipe as read from `train.py` (re-verify from its argparse defaults): 80 epochs, batch 5,
  LR 2e-4 decayed at epochs 45 and 65, EMA 0.999, class-weighted loss (`--weight_param 1.03`), VSCP augmentation.
