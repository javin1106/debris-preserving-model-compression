"""Efficiency benchmark (Phase 4.3, laptop only): params, conv/linear MACs, size, latency, peak memory.

Protocol: batch 1, input 1×11×240×240, FP32, 50 warm-up + 200 timed forwards, median and p90 (ms),
repeated ``--repeats`` times → median of medians. GPU timing with ``torch.cuda.synchronize()``;
CPU timing with ``torch.set_num_threads(n)`` for each n in ``--cpu_threads``. Never run on the H100.

CPU note: the official LightHam decoder builds its random NMF bases with ``torch.rand(...).cuda()``
unconditionally, so the model cannot run on CPU as written. For CPU timing only, this script swaps in a
``_build_bases`` that creates the same CPU random tensor and leaves it on the input's device
(in-process monkeypatch; no file under third_party/ is modified; GPU behaviour is unchanged).

MACs: counted with forward hooks on Conv2d / Linear (multiply-accumulates; ×2 for FLOPs). The NMF
matrix products inside the decoder are not counted (reported separately as ``nmf_bmm_macs``).

Usage: python -m src.debriskd.benchmark --widths 1.0 0.5 0.25 0.125 [--ckpt w=path ...]
"""
from __future__ import annotations

import argparse
import io
import os
import platform
import resource
import statistics
import time
from typing import Dict, List

import numpy as np
import torch

from .runlog import disable_tf32, env_info, make_run_id, run_dir, write_json
from .students import build_student, count_params


def patch_nmf_for_cpu() -> None:
    """Make NMF bases follow the input device (CPU support). Same CPU RNG draw as the original."""
    from mmseg.models.decode_heads import ham_head

    def _build_bases(self, B, S, D, R, cuda=False):
        bases = torch.rand((B * S, D, R))
        if cuda and torch.cuda.is_available() and getattr(self, "_on_cuda", True):
            bases = bases.cuda()
        return torch.nn.functional.normalize(bases, dim=1)

    orig_forward = ham_head._MatrixDecomposition2DBase.forward

    def forward(self, x, return_bases=False):
        self._on_cuda = x.is_cuda
        return orig_forward(self, x, return_bases)

    ham_head.NMF2D._build_bases = _build_bases
    ham_head._MatrixDecomposition2DBase.forward = forward


def count_macs(model: torch.nn.Module, x: torch.Tensor) -> Dict[str, int]:
    macs = {"conv_linear": 0}
    hooks = []

    def conv_hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        macs["conv_linear"] += int(out.numel() // out.shape[0] * k)

    def lin_hook(m, inp, out):
        macs["conv_linear"] += int(out.numel() // out.shape[0] * m.in_features)

    for mod in model.modules():
        if isinstance(mod, torch.nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    # NMF: (B·S, D, N) with N = H/8·W/8, rank R; per step ~ 3 bmm of D·N·R plus R² terms; report a rough count
    head = model.decode_head
    D = head.ham_channels
    N = (x.shape[-2] // 8) * (x.shape[-1] // 8)
    R = head.hamburger.ham.R
    steps = head.hamburger.ham.eval_steps
    macs["nmf_bmm_macs"] = int((steps * 4 + 2) * D * N * R)
    return macs


def time_forward(model, x, n_warm: int, n_timed: int, cuda: bool) -> List[float]:
    ts = []
    with torch.no_grad():
        for i in range(n_warm + n_timed):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if cuda:
                torch.cuda.synchronize()
            if i >= n_warm:
                ts.append((time.perf_counter() - t0) * 1000)
    return ts


def summarise(runs: List[List[float]]) -> Dict[str, float]:
    meds = [statistics.median(r) for r in runs]
    p90s = [float(np.percentile(r, 90)) for r in runs]
    return {"median_ms": statistics.median(meds), "p90_ms": statistics.median(p90s), "medians_ms": meds}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--widths", nargs="+", type=float, default=[1.0, 0.5, 0.25, 0.125])
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--timed", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cpu_threads", nargs="+", type=int, default=[8, 1])
    ap.add_argument("--no_gpu", action="store_true")
    ap.add_argument("--no_cpu", action="store_true")
    ap.add_argument("--subset", type=int, default=0, help="smoke test: use N timed runs")
    ap.add_argument("--epochs", type=int, default=None, help="unused")
    args = ap.parse_args()
    if args.subset:
        args.warmup, args.timed, args.repeats = 2, args.subset, 1

    disable_tf32()
    patch_nmf_for_cpu()
    torch.backends.cudnn.benchmark = False
    rid = make_run_id("p4", "bench", "laptop")
    rd = run_dir(rid)
    out = {"run_id": rid, "protocol": vars(args), "cpu": platform.processor() or platform.machine(),
           "cpu_count": os.cpu_count(), "models": {}}
    x_cpu = torch.randn(1, 11, 240, 240)
    for w in args.widths:
        torch.manual_seed(0)
        m = build_student(w).eval()
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        rec = {"width": w, "params": count_params(m), "state_dict_bytes": buf.getbuffer().nbytes,
               **count_macs(m, x_cpu)}
        if not args.no_gpu and torch.cuda.is_available():
            mg = m.cuda()
            xg = x_cpu.cuda()
            torch.cuda.reset_peak_memory_stats()
            runs = [time_forward(mg, xg, args.warmup, args.timed, True) for _ in range(args.repeats)]
            rec["gpu"] = {**summarise(runs), "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
                          "device": torch.cuda.get_device_name(0)}
            m = mg.cpu()
            torch.cuda.empty_cache()
        if not args.no_cpu:
            for n in args.cpu_threads:
                torch.set_num_threads(n)
                rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                runs = [time_forward(m, x_cpu, args.warmup, args.timed, False) for _ in range(args.repeats)]
                rec[f"cpu_{n}t"] = {**summarise(runs), "ru_maxrss_mb_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                                    "ru_maxrss_mb_before": rss0 / 1024}
        out["models"][str(w)] = rec
        print(w, {k: (v["median_ms"] if isinstance(v, dict) and "median_ms" in v else v) for k, v in rec.items()}, flush=True)
    write_json(rd / "metrics.json", out)
    write_json(rd / "env.json", {**env_info(), "torch_num_threads_default": torch.get_num_threads()})
    print("wrote", rd / "metrics.json")


if __name__ == "__main__":
    main()
