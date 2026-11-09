# Debris-Preserving Model Compression

How small can a satellite-image segmentation model get before it stops finding floating marine debris?

We take **MariNeXt**, a published model that labels every pixel of Sentinel-2 sea images into 15 classes
(water, debris, oil, ships, algae, …), and train smaller copies of it. We then check which classes the small
models lose first, and whether letting the big model "teach" the small one helps it keep debris.

College project, NIT Jalandhar — Javin Chutani, Jashan Aneja.

## The idea in short

- **Shrink:** same architecture, every layer made ½, ¼ or ⅛ as wide (4.2 M → 1.1 M → 0.32 M → 0.10 M parameters).
- **Teach:** the small model (student) learns from the human labels *and* from the big model's (teacher's) answers.
- **Use the unlabelled pixels:** in the MADOS dataset most pixels have no label. Where the teacher is confident,
  we let the student copy the teacher's soft answers on those pixels too. We never turn them into hard labels.
- **Compress further:** store weights as 8-bit integers (INT8) instead of 32-bit floats.

## Main results

Debris F1 (%), mean ± sd over 3 seeds (5 released checkpoints for the teacher). From
[results/headline.md](results/headline.md), which is generated from the run logs.

| model | parameters | size | val debris F1 | test debris F1 | CPU time (1 thread) |
|---|---|---|---|---|---|
| Teacher (original MariNeXt) | 4.24 M | 17.2 MB | 63.9 ± 3.3 | 48.5 ± 4.9 | 63.8 ms |
| ½ width, labels only | 1.13 M | 4.7 MB | 52.5 ± 5.5 | 41.2 ± 4.7 | 25.1 ms |
| ½ width, our method | 1.13 M | 4.7 MB | 60.4 ± 1.8 | 47.8 ± 3.0 | 25.1 ms |
| ½ width, our method + INT8 | 1.13 M | 1.9 MB | 58.2 ± 1.7 | 46.4 ± 3.4 | 22.1 ms |
| ¼ width, our method | 0.32 M | 1.5 MB | 42.0 ± 3.7 | 35.7 ± 3.9 | 10.7 ms |

What we found:

- At **½ width**, our method brings debris F1 close to the teacher's, with a model about 9× smaller (INT8) and about 2.5× faster on CPU.
- Simply weighting debris ×2 in the loss does as well on validation. Our method is better on the test set.
- At **¼ width**, all methods are within noise of each other. This is roughly where the model becomes too small.
- **Rare classes break first,** not only debris but also foam, waves, jellyfish and sargassum.
  Our method also makes a few classes (e.g. dense sargassum) worse. See [results/perclass.md](results/perclass.md).
- About 90% of the confident unlabelled pixels are water, so the extra guidance mostly teaches the student "what is not debris".
  See [results/teacher_reliability.md](results/teacher_reliability.md).

The full write-up is in [paper/draft.md](paper/draft.md).

## Repository layout

| folder | contents |
|---|---|
| `src/debriskd/` | our code: student models, distillation loss, training, evaluation, INT8, speed tests |
| `configs/settings/` | one file per training setting (B = labels only, C = debris ×2, D/E/F = with teacher) |
| `scripts/` | launch scripts and the scripts that build every table in `results/` |
| `results/` | all tables and figures (generated, never typed by hand) |
| `tests/` | unit tests (`pytest -q`) |
| `third_party/mados/` | official MariNeXt code (git submodule, unchanged; optional patches for newer library versions in `docs/PATCHES.md`) |
| `docs/` | project brief, phase plans, progress log (`STATUS.md`), environment notes |

`data/`, `checkpoints/` and `runs/` are not in git.

## Running it

**Setup** (Ubuntu / WSL2, NVIDIA GPU). This follows the official MariNeXt install (Python 3.8, PyTorch 1.11, mmcv-full 1.6.0, GDAL):

```bash
git clone --recursive https://github.com/javin1106/debris-preserving-model-compression.git
cd debris-preserving-model-compression
bash scripts/setup_env_orig.sh        # creates the conda env "mados"
conda activate mados
```

Download [MADOS](https://doi.org/10.5281/zenodo.10664073) and the five MariNeXt checkpoints (link in the
[official repo](https://github.com/gkakogeorgiou/mados)) into `checkpoints/teacher/<1-5>/`.

**Prepare the data once** (saves the official data loader's output for faster training):

```bash
python -m src.debriskd.data --path ~/mados_data/MADOS --cache ~/mados_data/cache
```

**Train a student** (½ width, our method, seed 0):

```bash
python -m src.debriskd.train --cache ~/mados_data/cache --config configs/settings/F2.yaml \
    --width 0.5 --seed 0 --phase p3
```

Add `--subset 20 --epochs 1` for a quick test. Each run writes its config, log and metrics to `runs/<run_id>/`.

**Evaluate, compress, time and summarise:**

```bash
python -m src.debriskd.evaluate  --help    # per-class scores on labelled pixels
python -m src.debriskd.quantize  --help    # INT8 via ONNX Runtime
python -m src.debriskd.benchmark --help    # size, FLOPs, GPU/CPU speed
python scripts/summarise_headline.py       # rebuilds results/headline.md
pytest -q
```

## Ground rules we followed

- Unlabelled pixels are never scored and never treated as "water".
- All choices were made on the validation split. The test split was only used for the final numbers.
- Every number in `results/` comes from a logged run in `runs/`.
- Speed was measured only on our laptop (RTX 4060 Laptop GPU and i7-13620H CPU).

## Credits

- MariNeXt and MADOS: Kikaki, Kakogeorgiou, Hoteit, Karantzalos (2024), *ISPRS J. Photogramm. Remote Sens.* 210, 39–54.
  Code: <https://github.com/gkakogeorgiou/mados> (MIT).
- SegNeXt: Guo et al. (2022). Knowledge distillation: Hinton, Vinyals, Dean (2015).
