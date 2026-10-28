# Environment notes

## Official repo (submodule)
- `third_party/mados` → https://github.com/gkakogeorgiou/mados
- Pinned commit: `c20a7e972bdf111126540d8e6caa45808352f138` (2024-06-14, "fix in evaluation")

## Laptop
- Windows 11 Home 10.0.26300; WSL2 distro `Ubuntu` (Ubuntu 26.04 LTS)
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, driver 560.94 (visible inside WSL via `nvidia-smi`)
- WSL sees 16 logical CPUs and ~7 GB RAM (WSL default = half of host RAM)

## env-orig (WSL2)
Script: `scripts/setup_env_orig.sh` (run inside WSL from the repo root). It installs Miniforge to
`~/miniforge3` (no conda was present) and then runs the README "Installation" commands verbatim:

```bash
conda create -n mados python=3.8.12
conda activate mados
conda install -c conda-forge gdal==3.3.2
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
conda install pytables==3.7.0
```

Second half (pip + pytables + checks): `scripts/setup_env_orig_pip.sh`. Full export: `docs/environment-orig.yml`.

**Result (2026-10-14):** torch 1.11.0+cu113, `torch.cuda.is_available() = True` on the RTX 4060 Laptop GPU,
mmcv-full 1.6.0, timm 0.4.12, rasterio 1.3a3, pytables 3.7.0, GDAL 3.3.2. The official `MariNext` wrapper imports
from the submodule, runs a forward pass on CUDA (1×11×240×240 → 1×15×60×60), and released checkpoint 1 loads with
all keys matched (after `evaluation.py`'s key remap).

**Workarounds (no package version changed):**
1. *Slow / dropping network:* the 1.6 GB torch wheel timed out in pip. It is now downloaded first with resumable
   `curl -C -` into `~/wheels` and pip is given `-f ~/wheels`. Same wheel, same version.
2. *`set -u` vs GDAL's conda activation hook:* `envs/mados/etc/conda/activate.d/gdal-activate.sh` reads unset
   variables, so `conda activate` must run with `set +u` inside scripts.
3. *Ubuntu 26.04 glibc 2.43 vs PyTorch 1.11:* `import torch` failed with
   `libtorch_cpu.so: cannot enable executable stack as shared object requires: Invalid argument`
   (glibc ≥ 2.41 refuses libraries that request an executable stack). Fixed without touching any binary by setting
   the glibc tunable as a conda env variable:
   `conda env config vars set GLIBC_TUNABLES=glibc.rtld.execstack=2` (re-activate the env after setting it).
   The env will also need this on any other machine with glibc ≥ 2.41.
4. `conda install gdal==3.3.2` (README step) upgrades Python 3.8.12 → 3.8.15 as part of its solve; we keep the
   README's behaviour.
5. RTX 4060 is compute capability 8.9; the cu113 wheel has no sm_89 kernels but runs via sm_86 binary
   compatibility (verified with a CUDA matmul and a MariNeXt forward pass).

WSL memory raised for the full-split loaders: `C:\Users\LENOVO\.wslconfig` → `[wsl2] memory=12GB, swap=8GB`
(user approved 2026-10-14; effective after `wsl --shutdown`).

**Added 2026-10-29 (user approved, Phase 4 INT8):** `pip install onnx==1.17.0 onnxruntime==1.19.2` in env-orig
(last releases with Python 3.8 wheels; pulled only small deps: coloredlogs, flatbuffers, humanfriendly, mpmath, sympy;
no pinned package changed — torch 1.11.0+cu113, numpy 1.23.1 as before). ONNX Runtime is CPU-only here.

**WSL quirks found 2026-10-28:** (a) the WSL login shell `cd`s to `~/zulip` (user's profile), so launchers must
`cd` inside the command, not rely on `wsl --cd`; (b) Ubuntu 26.04 ships uutils `tail`, whose `-F` misses writes to
files under `/mnt/c` — use `tail --use-polling`; (c) after a sleep/wake the NVIDIA driver logged ~1,200
`nvlddmkm` FECS exceptions and killed CUDA processes ("CUDA error: unknown error") — a Windows restart cleared it.

## Data
- MADOS: Zenodo record 10664073, single file `MADOS.zip`, 4.04 GB, md5 `1076b25d2797be3095d82a61105c7380`.
- Location: `data/MADOS/` (gitignored): 174 `Scene_*` folders + `splits/{train,val,test}_X.txt`, 56,760 zip entries.
  Downloaded with resumable parallel range requests (Zenodo is slow per connection); md5 verified = Zenodo's.
  Extracted with Python `zipfile` inside WSL.
- Loading cost through the official loader from `/mnt/c` (WSL → NTFS): `scripts/inspect_data.py` (all 3 splits,
  sequentially) took 31 min wall clock, peak RSS 8.1 GB. Expect several minutes of loading before any train/eval run.
- Checkpoint location: `evaluation.py` takes `--model_path <folder>`, so `checkpoints/teacher/<k>/` is passed directly;
  no `marinext/trained_models` symlink is needed.
- Preprocessing: **none is required by the official train/eval code.** `utils/dataset.py` reads the
  per-band, per-resolution GeoTIFFs (`<tile>/<10|20|60>/*L2R_rhorc*_<crop>.tif`) directly and upsamples
  20 m / 60 m bands to 10 m with nearest-neighbour at load time. `utils/stack_patches.py` writes a separate
  stacked copy (`<path>_nearest/`) that is only used by `utils/spectral_extraction.py`. We do not run it.

## Checkpoints
Downloaded 2026-10-14 from the README's Google Drive folder (`1VwkFp47TEvRVXHNbucBmmylfZwIUmCWx`) with
`python -m gdown --folder --continue <url> -O checkpoints/teacher` (gdown run on Windows). One file per folder,
matching the README usage `--model_path marinext/trained_models/<k>`.

| file | bytes | SHA-256 |
|---|---|---|
| `checkpoints/teacher/1/marinext_1.pth` | 17,184,373 | `b41ab9d9dd5b200adadecc48b9e66821d854590ecd8aa059142570fb1151be62` |
| `checkpoints/teacher/2/marinext_2.pth` | 17,184,373 | `57e0043c32551af9d57019380a995cec4d2082b3cf8698873b21e8144aac3712` |
| `checkpoints/teacher/3/marinext_3.pth` | 17,184,373 | `9c244687bed8f2d0ab70cd4f08d2a73b8e89831815f0a78518461b657fac9746` |
| `checkpoints/teacher/4/marinext_4.pth` | 17,184,373 | `78d7b1f83084354136056a69ef694d64257cb39b3dcf24396b68c84d5ca987a9` |
| `checkpoints/teacher/5/marinext_5.pth` | 17,184,373 | `a7cc5d508656b1b7c3759c1e6fe00f8fd9066ef19d585a0137e4410ba9621043` |

## Official recipe (read from `marinext/train.py` at the pinned commit)

### argparse defaults
| flag | default | note |
|---|---|---|
| `--mode` | `train` | |
| `--epochs` | 80 | |
| `--batch` | 5 | before VSCP (see below) |
| `--resume_from_epoch` | 0 | loads `checkpoint_path/<epoch>/model.pth`; optimizer/scheduler/EMA state are **not** restored |
| `--input_channels` | 11 | |
| `--output_channels` | 15 | |
| `--weight_param` | 1.03 | |
| `--vscp` | True | |
| `--label_smoothing` | 0.0 | |
| `--clip_grad` | None | |
| `--lr` | 2e-4 | Adam |
| `--decay` | 0 | Adam weight decay |
| `--reduce_lr_on_plateau` | 0 | → MultiStepLR |
| `--lr_steps` | `[45,65]` | gamma 0.1 |
| `--checkpoint_path` | `marinext/trained_models` | |
| `--eval_every` | 1 | |
| `--model_ema` | True | |
| `--model_ema_decay` | 0.999 | |
| `--model_ema_eval` | True | |
| `--num_workers` | 0 | |
| `--pin_memory` | False | |
| `--prefetch_factor` | 2 | |
| `--persistent_workers` | False | |
| `--tensorboard` | `tsboard_segm` | |

### Details
- **Seed:** `seed_all(0)` is hard-coded at the start of `main()` (torch, cuda, numpy, random;
  `cudnn.deterministic=True`, `cudnn.benchmark=False`). There is **no `--seed` flag.** `utils/dataset.py` also
  seeds 0 at import. `seed_all(0)` is called again before every validation pass.
- **Loss:** `CrossEntropyLoss(ignore_index=-1, reduction='mean', weight=w)` with
  `w = 1 / ln(1.03 + class_distr)`; `class_distr` is a hard-coded 15-vector in `utils/dataset.py`
  (pixel-level class frequencies, sum ≈ 1).
- **Augmentation:** in `MADOS.__getitem__` (train only): random flip (one of h/v/both/none, with "none" ×2)
  then, with p = 0.8, rotation by k·90°. Then **VSCP** in the training loop: with batch 5,
  `n_augmented = 5 // 2 = 2`; for i in {0, 1}, every labelled pixel of image i+2 (image bands and label) is pasted
  onto image i. The 2 new images are concatenated to the batch → **7 images per step**. Unlabelled pixels of the
  source image are not pasted.
- **Logits:** decoder output is upsampled to 240×240 with `F.upsample(..., mode='bilinear')` (align_corners default False).
- **EMA:** `timm.utils.ModelEma`, updated after every optimizer step. Its decay follows
  `cosine_scheduler(0.999, 0.999, ...)`, i.e. constant 0.999. EMA weights are saved each eval epoch as
  `trained_models/<epoch>/model_ema.pth`; raw weights as `trained_models/<epoch>/model.pth`.
- **LR schedule:** `MultiStepLR.step()` is called once per evaluation (every epoch with `eval_every=1`),
  so milestones 45 and 65 are in epochs.
- **Model selection:** the script trains all 80 epochs and saves every epoch; it **does not select a best epoch**.
  It only logs val metrics (raw and EMA) per epoch. `evaluation.py`'s default `--model_path`
  is `trained_models/45/model_ema.pth`, which hints that EMA weights from a chosen epoch were used, but how the
  released checkpoints were chosen is not stated in code or README.
- Requires CUDA in the training loop (`image.cuda()` is hard-coded).

## Official evaluation (read from `marinext/evaluation.py`)
- Defaults: `--split test`, `--test_time_augmentations True`, `--batch 1`, `--results_percentage True`,
  `--predict_masks False`. **TTA is on by default.**
- TTA (`utils/test_time_aug.py`): 8 views (rot 0/90/180/270 × with/without hflip); each view's argmax map is
  inverted back and the 8 maps are combined by **per-pixel majority vote (`torch.mode`)** of hard labels, not by
  averaging probabilities.
- `--model_path` is a **folder**; every `*.pth` in it is loaded. Checkpoint keys are remapped
  `decoder → decode_head`, and keys containing `proj1` / `proj2` are dropped before loading — the released
  checkpoints therefore contain keys the current `MariNext` class does not have, i.e. they were produced by a
  code version different from the published `train.py`.
- Multi-model ensembling is attempted (`torch.mode` over models) but the result `all_predictions` is never used;
  the metric is computed from `predictions` of the **last** model in the glob. With one `.pth` per folder
  (the README usage: `--model_path marinext/trained_models/1`) this is a single-model evaluation.
- Metrics: predictions and targets flattened, pixels with target == -1 removed, then sklearn on the remaining
  pixels: per-class precision/recall/F1/IoU from `confusion_matrix()`, plus macro/micro/weighted P/R/F1,
  overall accuracy, macro IoU (`jaccard_score(average='macro')`). Aggregation is over **all pixels of the split
  pooled together** (not per-image averages).
- How the 5 released checkpoints were produced: not stated in the README or code (README only says
  "5 different runs").
