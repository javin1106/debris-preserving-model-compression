# Phase 0 — Setup (local laptop)

## Goal
A working repository, the original environment (`env-orig`) running on the laptop GPU,
MADOS and the five pretrained MariNeXt checkpoints downloaded, and a factual report on the data.
No training in this phase.

## Before you start
Ask the user: laptop OS (Windows → WSL2), and how the institute H100 is accessed.
Record both answers in `docs/STATUS.md`.

## Tasks

### 0.1 Repository skeleton
- `git init` (if needed); create the layout from `CLAUDE.md`.
- Add the official repo as a submodule at `third_party/mados`:
  `git submodule add https://github.com/gkakogeorgiou/mados third_party/mados`
  Record the pinned commit hash in `docs/ENVIRONMENT.md`.
- `.gitignore`: `data/`, `checkpoints/`, `runs/`, `*.pth`, `*.onnx`, `__pycache__/`, `.ipynb_checkpoints/`.
- Create `src/debriskd/__init__.py`, `tests/`, `configs/`, `scripts/`, `results/`.
- Create empty `docs/PATCHES.md` and `docs/ENVIRONMENT.md`.

### 0.2 Original environment (`env-orig`)
- Read `third_party/mados/README.md` and `requirements.txt`. Follow the README's install commands exactly
  (conda, Python 3.8, PyTorch 1.11 + CUDA 11.3, mmcv-full 1.6.0, timm 0.4.12, GDAL).
- On Windows: do this inside WSL2 Ubuntu; check `nvidia-smi` works inside WSL first.
- Verify: `python -c "import torch, mmcv; print(torch.__version__, torch.cuda.is_available(), mmcv.__version__)"`
  and that the MariNeXt wrapper imports from the submodule.
- Export: `conda env export > docs/environment-orig.yml`.
- Write the exact commands that worked, and any workaround, into `docs/ENVIRONMENT.md`.

### 0.3 Data and checkpoints
- Download MADOS from Zenodo (doi:10.5281/zenodo.10664073) into `data/MADOS/`.
- Run the official preprocessing exactly as the README describes (e.g. `utils/stack_patches.py`);
  record the command and output location in `docs/ENVIRONMENT.md`.
- Download the 5 pretrained checkpoints (Google Drive link in the README) into `checkpoints/teacher/`;
  if the official scripts expect `marinext/trained_models/`, use a symlink rather than editing their code.
- Record file names and sizes; compute SHA-256 of each checkpoint into `docs/ENVIRONMENT.md`.

### 0.4 Data report — `scripts/inspect_data.py` → `results/data_report.md`
Read the official dataset loader (`third_party/mados/utils/dataset.py`) and report, from the data itself:
- class names and their integer indices (which index is marine debris)
- the ignore-index value and how unlabelled pixels are encoded on disk
- number of bands, band order, patch size, dtype, normalisation applied by the loader
- number of patches per split (train / val / test)
- labelled-pixel count per class per split, and fraction of pixels that are unlabelled
- what the VSCP augmentation does to images and labels (read `utils/vscp.py`)

### 0.5 Parameter count — `src/debriskd/count_params.py`
- Build MariNeXt through the official wrapper with the official config.
- Print total and trainable parameters; compare with the analytic value 4,238,063.
- Write the result into `results/data_report.md` (section "Model size").

### 0.6 Read the official recipe
Read `third_party/mados/marinext/train.py` and `evaluation.py`. In `docs/ENVIRONMENT.md` record:
all argparse defaults, how class weights are computed from `--weight_param`, where EMA is applied,
how evaluation aggregates metrics, whether TTA is on by default, and how the 5 checkpoints were produced (if stated).
If any value differs from `CLAUDE.md` "Key facts", log it under "Corrections" in `docs/STATUS.md`.

## Deliverables
- Repo skeleton committed; submodule pinned.
- `docs/ENVIRONMENT.md`, `docs/environment-orig.yml`
- `results/data_report.md`
- `src/debriskd/count_params.py`, `scripts/inspect_data.py`

## Acceptance criteria
- [ ] `env-orig` imports torch, mmcv and the MariNeXt wrapper; CUDA available on the laptop GPU.
- [ ] Data preprocessed with the official script; all three splits load through the official loader.
- [ ] 5 checkpoints present with recorded hashes.
- [ ] `results/data_report.md` states the debris class index and the ignore index **as read from data/code**.
- [ ] Parameter count measured and compared to 4,238,063.
- [ ] `docs/STATUS.md` updated; Phase 0 checklist item ticked by Claude Code only after all of the above.

## Pitfalls
- Don't "fix" the official code to make it run on newer libraries here — that's Phase 1b on the H100.
- Don't guess the ignore index or class order; read them.
