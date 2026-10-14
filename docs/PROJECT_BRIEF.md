# Project brief — Debris-Preserving Model Compression for Satellite Image Segmentation

## One sentence
Find the smallest MariNeXt student that keeps marine debris detectable in Sentinel-2 imagery,
and test whether teacher guidance on *unlabelled* pixels pushes that limit further.

## Why it matters
- Floating debris covers a tiny share of pixels; when a model shrinks, rare classes like debris
  tend to collapse before common classes like water.
- MADOS labels are sparse: most pixels are unlabelled (ignore index) and give no supervision.
- On-board satellite processing would let a satellite downlink only scenes with debris,
  but on-board hardware runs at single-digit watts (e.g. ESA Φ-Sat-1 ran a CNN on a ~2 W VPU;
  FPGA accelerators are the other route). The Edge SpAIce project (CERN hls4ml toolchain) reports
  FPGA deployment of ~100k-parameter models today, aiming at 0.5–1 M.
- Related work to acknowledge: Edge SpAIce already distils MariNeXt for FPGA deployment.
  Our distinct contribution: confidence-weighted distillation on ignored pixels + per-class
  analysis of where debris accuracy collapses under compression.

## Models
- **Teacher:** MariNeXt (SegNeXt-Tiny style), official config, ≈4.24 M parameters. Frozen during student training.
- **Students:** same architecture with every channel width scaled by m ∈ {½, ¼, ⅛}
  (embed dims 32-64-160-256 × m; decoder channels 256 × m; depths, MLP ratios, 11 input bands, 15 classes unchanged).
- **Other compression routes:** structured channel pruning of the teacher (25%, 50%) + fine-tune;
  INT8 quantization (post-training static via ONNX Runtime; QAT only if accuracy drops too far).

## Loss (student training)
For a batch, teacher logits t and student logits s are computed on the **same augmented input**
(teacher run live, `torch.no_grad()`, eval mode). Let `lab` = pixels with a human label, `ign` = ignored pixels.

- Softened distributions: p_t = softmax(t / T), p_s = softmax(s / T)
- Per-pixel KD: kl(x) = T² · Σ_c p_t,c · (log p_t,c − log p_s,c)
- Teacher confidence: c(x) = max_c softmax(t)_c   (at T = 1)
- Confidence weight: w(x) = clamp((c(x) − τ) / (1 − τ), 0, 1)
- Debris weight: d(x) = debris_w if argmax teacher = debris class else 1 (clipped, debris_w ≤ debris_w_max)

L_total = L_sup  +  λ_L · mean_{lab}(kl)  +  λ_U · mean_{ign, w>0}(w · d · kl)

- L_sup = the official supervised loss (cross-entropy with the official class weighting, ignore index respected).
- λ_U warms up linearly from 0 over the first few epochs.
- Teacher outputs on ignored pixels are evidence, never ground truth. Never convert them to hard labels.

## Experiment settings (identical recipe, split, hardware)
| ID | Model | Change | What it proves |
|---|---|---|---|
| A | Teacher | MariNeXt, full size | Upper reference; teacher reliability |
| B | Student | Official recipe only (sparse labels) | Cost of compression |
| C | Student | B + extra debris-focused CE reweighting | Is simple imbalance handling enough? |
| D | Student | B + KD on labelled pixels (λ_U = 0) | Value of standard KD |
| E | Student | B + KD on ignored pixels only (λ_L = 0, debris_w = 1) | Value unique to sparse labels |
| F | Student | B + both KD terms + debris weighting | Full proposed method |
| G | Pruned teacher | 25% / 50% channels removed + fine-tune | Alternative compression route |
| H | INT8 | 8-bit weights (teacher and best student) | Precision vs size trade-off |

Note: the official recipe already contains class weighting (`--weight_param`). So B keeps it,
and C adds an *additional* debris-focused reweighting. Confirm this definition with the user in Phase 2.

Key comparisons: B→C, B→D, D→E, E→F, each at the same student width, mean ± sd over 3 seeds.

## Metrics
- Accuracy (labelled pixels only): per-class precision, recall, F1, IoU; mIoU; macro-F1.
  Headline: **debris F1, recall, IoU**.
- Efficiency: parameters, file size on disk (FP32 state dict, ONNX FP32, ONNX INT8), FLOPs at 1×11×240×240,
  latency (batch 1, 50 warm-up + 200 timed runs, median and p90), peak memory.
  Measured on the local RTX 4060 Laptop GPU and CPU only.

## Expected outputs
1. Debris-F1 versus model-size and versus latency curves.
2. Per-class degradation table (teacher → each compressed model): which classes break first.
3. Settings B–F comparison at 1–2 widths near the cliff (mean ± sd).
4. Recommended deployable model with measured size, latency and memory.
A negative result (e.g. KD does not beat class weighting) is a valid finding and must be reported honestly.

## Alternatives considered (already decided)
- Hard pseudo-labels on all unlabelled pixels — rejected (confirmation bias; water dominates).
- Foundation model + LoRA — rejected (far too large for on-board hardware).
- Feature-level KD; real FPGA latency via hls4ml — future work, out of scope.

## Hardware
- Local: RTX 4060 Laptop GPU 8 GB (Windows 11 + WSL2), CPU and RAM as recorded in `docs/ENVIRONMENT.md` —
  development, smoke tests, exact reproduction, latency benchmarks.
  (The original brief named an RTX 3050 6 GB; corrected 2026-10-14, see `docs/STATUS.md` → Corrections.)
- Institute: NVIDIA H100 — long training runs, accessed over SSH.

## References
1. Kikaki, K., Kakogeorgiou, I., Hoteit, I., Karantzalos, K. (2024). Detecting Marine Pollutants and Sea Surface
   Features with Deep Learning in Sentinel-2 Imagery. ISPRS J. Photogramm. Remote Sens. 210, 39–54.
2. Official code: https://github.com/gkakogeorgiou/mados
3. MADOS dataset: doi:10.5281/zenodo.10664073
4. Guo et al. (2022). SegNeXt. NeurIPS.
5. Hinton, Vinyals, Dean (2015). Distilling the Knowledge in a Neural Network.
6. Fang et al. (2023). DepGraph: Towards Any Structural Pruning. CVPR. (Torch-Pruning library)
7. Edge SpAIce project, NGT presentation, CERN Indico.
