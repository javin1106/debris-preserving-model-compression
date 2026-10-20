# STATUS

## Current phase
**Phase 1 — Teacher reproduction (local) + port to H100** → `docs/phases/phase-1-teacher.md`
(advanced on 2026-10-15 at the user's instruction: "do as u feel is okay, just do")

(The user changes this line to move to the next phase. Claude Code never advances it on its own.)

## Phase checklist
- [x] Phase 0 — Setup (repo, env-orig, data, checkpoints, data report)
- [ ] Phase 1 — Teacher reproduction (local) + port to H100   · GATE  (1a done, GATE 1 passed; 1b/1c need the H100)
- [ ] Phase 2 — Student builder + width sweep                  · GATE: user picks widths
- [ ] Phase 3 — Distillation experiments (settings C–F)
- [ ] Phase 4 — Pruning, INT8, benchmarking
- [ ] Phase 5 — Analysis, figures, report

## Machines
- Laptop: Windows 11 Home (10.0.26300), WSL2 Ubuntu installed → `env-orig` goes in WSL2.
  GPU: NVIDIA GeForce RTX 4060 Laptop, 8 GB, driver 560.94 (Windows `nvidia-smi`, CUDA 12.6 driver API).
- H100: accessed by SSH to a server → launch helpers use tmux/nohup.

## Open questions for the user
- H100 SSH details (host, GPU count, how many parallel jobs, storage path) — needed for Phase 1b.

## Decisions
<!-- date — decision — reason -->
- 2026-10-14 — Latency benchmarks and `env-orig` run on the laptop's RTX 4060 Laptop GPU (8 GB) — it is the GPU the team actually has; user confirmed.
- 2026-10-14 — H100 jobs launched over SSH with tmux/nohup scripts — user's access method.
- 2026-10-14 — WSL2 memory raised to 12 GB + 8 GB swap (`.wslconfig`) — official loader peaked at 8.1 GB RSS; user approved.
- 2026-10-15 — Our `train.py` gets `--seed` (default 0). With seed 0 it is identical to the official script (which hard-codes `seed_all(0)` and `Generator().manual_seed(0)`); the seed replaces those two. The re-seed before each validation pass also uses `--seed` (official: `seed_all(0)`), so seed 0 reproduces the official RNG behaviour exactly and other seeds are not reset to 0 every epoch. — Phases 1c/2/3 need seeds 0, 1, 2; user delegated the choice.
- 2026-10-15 — Model selection rule for all models we train: EMA weights (`model_ema.pth`) from the epoch with the best **validation macro-F1** (official per-epoch val log); ties → earlier epoch. Debris-F1 not used because val has only 1,162 debris px (noisy selection, and it would favour debris-targeted settings). The same rule for teacher, students, pruned models. — Official script has no rule; user delegated the choice.
- 2026-10-15 — Setting definitions (user delegated): **B** = official recipe exactly; **C** = B with the debris class weight in the official weighted CE multiplied by `extra_debris_ce_weight` (tune {2, 4} on val). Configs in `configs/settings/{B..F}.yaml`.
- 2026-10-15 — Provisional KD teacher = released checkpoint 5 (`checkpoints/teacher/5/marinext_5.pth`): best **validation** debris F1 of the 5 released checkpoints (single 69.1, TTA 73.2; `results/teacher_repro.md`). To be re-checked against our retrained teacher (Phase 1c). Test scores not used.
- 2026-10-15 — KD teacher forward wrapped in `torch.random.fork_rng`: MariNeXt's decoder draws random NMF bases on every forward; isolating them keeps the student's RNG stream identical to setting B, so B vs D/E/F differ only by the loss.
- 2026-10-15 — User: skip the H100 for now and run on the laptop. Width sweep (B, seed 0, widths ½ ¼ ⅛) and one teacher retrain (seed 0) run on the RTX 4060 in env-orig, 2 jobs at a time (each CUDA-11.3 process holds ~4.1 GB host RAM; 3–4 parallel jobs swapped heavily). Launch: `scripts/run_local_jobs.sh configs/jobs/pair{1,2}.txt`.
- 2026-10-14 — env-orig on glibc 2.43 uses `GLIBC_TUNABLES=glibc.rtld.execstack=2` (conda env var) so PyTorch 1.11 loads; no package versions changed.

## Corrections (docs vs actual code)
<!-- date — what the docs said — what the code actually does -->
- 2026-10-14 — Docs said local GPU is RTX 3050 6 GB — `nvidia-smi` shows RTX 4060 Laptop 8 GB. CLAUDE.md and PROJECT_BRIEF.md updated; phase files that say "RTX 3050" / "3050" mean this RTX 4060.
- 2026-10-14 — Phases 1c/2/3 assume runs with seeds 0, 1, 2 — official `train.py` hard-codes `seed_all(0)` and has no `--seed` flag. Running other seeds needs a (small) change to the recipe code → ask the user before Phase 1c.
- 2026-10-14 — Phase 2.3 says "keep model selection exactly as the official script does it (e.g. best validation epoch)" — official `train.py` does **no** model selection: it trains 80 epochs, saves raw + EMA weights every epoch and only logs val metrics. A selection rule must be decided (Phase 1c/2).
- 2026-10-14 — Phase 0.3 suggests preprocessing with `utils/stack_patches.py` — the official loader reads the unstacked per-band TIFFs directly; stacking is only used by `spectral_extraction.py`. No preprocessing is needed for train/eval.
- 2026-10-14 — (code fact, affects Phase 1) `evaluation.py` loads every `.pth` in `--model_path` and builds a majority-vote ensemble, but then scores only the **last** model's predictions (`predictions` instead of `all_predictions`). With one checkpoint per folder (README usage) it is a single-model evaluation. TTA is on by default (8 views, hard-label majority vote).
- 2026-10-15 — (code fact) PyTorch 1.11 enables TF32 by default, so official `evaluation.py` runs with TF32 on (RTX 4060 supports it). Our evaluator with TF32 on reproduces it exactly (12 digits); with TF32 off (our rule) a few pixels flip (≤ 0.3 pts per class on ckpt 1).
- 2026-10-15 — (code fact, affects Phase 4) `ham_head.py` builds the NMF bases with `torch.rand(...).cuda()` unconditionally → MariNeXt cannot run on CPU as written. CPU benchmarking / ONNX export will need a documented patch (bases on the input's device).
- 2026-10-15 — (code fact) The LightHam decoder draws random NMF bases on every forward, also in eval mode; outputs depend on the RNG state. Official evaluation re-seeds 0 before every patch; we do the same.
- 2026-10-14 — (code fact) `evaluation.py` renames `decoder→decode_head` and drops `proj1`/`proj2` keys when loading — the released checkpoints were made with a code version different from the published `train.py`.

## Session log
<!-- date — phase — what was done — what is blocked — next step -->
- 2026-10-14 — Phase 0 — Wrote docs into repo; skeleton + `.gitignore`; `git init`; submodule pinned at c20a7e9; read README/train/eval/dataset/vscp/metrics/TTA and recorded the recipe in `docs/ENVIRONMENT.md`; wrote `src/debriskd/count_params.py` and `scripts/inspect_data.py` (not yet run). Started: MADOS download (4.04 GB, slow network ≈0.7 MB/s), env-orig install in WSL (first attempt killed by a WSL VM restart, re-running). — Blocked on downloads; checkpoints (Google Drive) not yet downloaded. — Next: verify env-orig, download checkpoints, unzip data, run `inspect_data.py`.
- 2026-10-14 — Phase 0 — Finished: env-orig works on the RTX 4060 (torch 1.11+cu113, mmcv-full 1.6.0; glibc workaround); MADOS downloaded (md5 OK) and unzipped; 5 checkpoints downloaded + SHA-256 recorded, checkpoint 1 loads with all keys matched; `results/data_report.md` generated (debris = index 0, ignore = -1, 1433/642/728 patches, ~99% pixels unlabelled); parameter count 4,238,063 = analytic. All Phase 0 acceptance criteria met. — Nothing blocked. — Next: user moves to Phase 1. Note for Phase 1: test split has only 973 debris pixels in 124 patches, so debris metrics will be noisy.
- 2026-10-15 — Phase 1a — `evaluate.py` (+ `metrics.py`, tests); all 5 released checkpoints on val/test, single + TTA (`results/teacher_repro.md`). **GATE 1 PASSED**: mean of 5, test, TTA: debris F1 51.32 ± 4.08, recall 70.42 ± 3.06, IoU 34.60 ± 3.76 vs paper 51.3 / 70.4 / 34.6 (paper = mean of 5 with TTA per project page). Cross-check vs official `evaluation.py`: exact with TF32 at torch default. Phase 1b prepared but not run (env-h100 script + focal-loss patch; no H100 yet). Phase 2/3 code written ahead (user asked for substantial work): `students.py` (exact param counts), `train.py` (+ resume, selection, KD/C options), `kd_loss.py`, configs B–F, `data.py` cache, `summarise_sweep.py`; 34 tests pass. Width sweep + teacher retrain launched on the laptop. — Blocked: H100 access for 1b/1c. — Next: summarise sweep when runs finish; smoke-test C–F; H100 details from user.
- 2026-10-20 — Phase 2/3 (laptop) — Width sweep B seed 0 finished for ×½ ×¼ ×⅛ (`results/width_sweep.md`): val debris F1 63.9 (released teacher mean) → 41.5 / 29.8 / 28.5; common classes barely drop at ×½ while debris drops 22 pts. ×¼ and ×⅛ selected epoch 80 (still improving at the end). Our full-size retrain (p1-A-w1.0-s0) stopped by the user before it started; optional, can run on H100. Laptop efficiency benchmark `p4-bench-laptop-20261020-1814` (`results/efficiency.md`): CPU 1-thread 84.6 → 36.7 → 20.8 → 12.5 ms; GPU latency does not fall below ~10 ms (launch-bound). Smoke tests B–F pass (`scripts/smoke_settings.sh`): losses finite, C doubles the debris CE weight, λU warm-up works; ~62% of ignored px pass τ = 0.8 and the teacher calls ~93% of those Marine Water. **Open:** B vs D(λ=0) train losses differ at the 5th digit (2.68574 vs 2.68567) — not bit-identical as designed; likely cuDNN algorithm choice under different free GPU memory, not RNG leakage — verify before claiming equivalence. Code pushed to github.com/javin1106/debris-preserving-model-compression (private). — Next: pick widths (user gate), H100 access.
