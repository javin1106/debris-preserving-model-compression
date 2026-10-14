# STATUS

## Current phase
**Phase 0 — Setup** → `docs/phases/phase-0-setup.md`

(The user changes this line to move to the next phase. Claude Code never advances it on its own.)

## Phase checklist
- [x] Phase 0 — Setup (repo, env-orig, data, checkpoints, data report)
- [ ] Phase 1 — Teacher reproduction (local) + port to H100   · GATE
- [ ] Phase 2 — Student builder + width sweep                  · GATE: user picks widths
- [ ] Phase 3 — Distillation experiments (settings C–F)
- [ ] Phase 4 — Pruning, INT8, benchmarking
- [ ] Phase 5 — Analysis, figures, report

## Machines
- Laptop: Windows 11 Home (10.0.26300), WSL2 Ubuntu installed → `env-orig` goes in WSL2.
  GPU: NVIDIA GeForce RTX 4060 Laptop, 8 GB, driver 560.94 (Windows `nvidia-smi`, CUDA 12.6 driver API).
- H100: accessed by SSH to a server → launch helpers use tmux/nohup.

## Open questions for the user
- Confirm definition of setting C (Phase 2).
- Before Phase 1c: add a `--seed` flag to our copy of `train.py`, and pick a model-selection rule (official script has none).

## Decisions
<!-- date — decision — reason -->
- 2026-10-14 — Latency benchmarks and `env-orig` run on the laptop's RTX 4060 Laptop GPU (8 GB) — it is the GPU the team actually has; user confirmed.
- 2026-10-14 — H100 jobs launched over SSH with tmux/nohup scripts — user's access method.
- 2026-10-14 — WSL2 memory raised to 12 GB + 8 GB swap (`.wslconfig`) — official loader peaked at 8.1 GB RSS; user approved.
- 2026-10-14 — env-orig on glibc 2.43 uses `GLIBC_TUNABLES=glibc.rtld.execstack=2` (conda env var) so PyTorch 1.11 loads; no package versions changed.

## Corrections (docs vs actual code)
<!-- date — what the docs said — what the code actually does -->
- 2026-10-14 — Docs said local GPU is RTX 3050 6 GB — `nvidia-smi` shows RTX 4060 Laptop 8 GB. CLAUDE.md and PROJECT_BRIEF.md updated; phase files that say "RTX 3050" / "3050" mean this RTX 4060.
- 2026-10-14 — Phases 1c/2/3 assume runs with seeds 0, 1, 2 — official `train.py` hard-codes `seed_all(0)` and has no `--seed` flag. Running other seeds needs a (small) change to the recipe code → ask the user before Phase 1c.
- 2026-10-14 — Phase 2.3 says "keep model selection exactly as the official script does it (e.g. best validation epoch)" — official `train.py` does **no** model selection: it trains 80 epochs, saves raw + EMA weights every epoch and only logs val metrics. A selection rule must be decided (Phase 1c/2).
- 2026-10-14 — Phase 0.3 suggests preprocessing with `utils/stack_patches.py` — the official loader reads the unstacked per-band TIFFs directly; stacking is only used by `spectral_extraction.py`. No preprocessing is needed for train/eval.
- 2026-10-14 — (code fact, affects Phase 1) `evaluation.py` loads every `.pth` in `--model_path` and builds a majority-vote ensemble, but then scores only the **last** model's predictions (`predictions` instead of `all_predictions`). With one checkpoint per folder (README usage) it is a single-model evaluation. TTA is on by default (8 views, hard-label majority vote).
- 2026-10-14 — (code fact) `evaluation.py` renames `decoder→decode_head` and drops `proj1`/`proj2` keys when loading — the released checkpoints were made with a code version different from the published `train.py`.

## Session log
<!-- date — phase — what was done — what is blocked — next step -->
- 2026-10-14 — Phase 0 — Wrote docs into repo; skeleton + `.gitignore`; `git init`; submodule pinned at c20a7e9; read README/train/eval/dataset/vscp/metrics/TTA and recorded the recipe in `docs/ENVIRONMENT.md`; wrote `src/debriskd/count_params.py` and `scripts/inspect_data.py` (not yet run). Started: MADOS download (4.04 GB, slow network ≈0.7 MB/s), env-orig install in WSL (first attempt killed by a WSL VM restart, re-running). — Blocked on downloads; checkpoints (Google Drive) not yet downloaded. — Next: verify env-orig, download checkpoints, unzip data, run `inspect_data.py`.
- 2026-10-14 — Phase 0 — Finished: env-orig works on the RTX 4060 (torch 1.11+cu113, mmcv-full 1.6.0; glibc workaround); MADOS downloaded (md5 OK) and unzipped; 5 checkpoints downloaded + SHA-256 recorded, checkpoint 1 loads with all keys matched; `results/data_report.md` generated (debris = index 0, ignore = -1, 1433/642/728 patches, ~99% pixels unlabelled); parameter count 4,238,063 = analytic. All Phase 0 acceptance criteria met. — Nothing blocked. — Next: user moves to Phase 1. Note for Phase 1: test split has only 973 debris pixels in 124 patches, so debris metrics will be noisy.
