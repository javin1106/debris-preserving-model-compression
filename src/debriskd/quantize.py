"""INT8 post-training static quantization with ONNX Runtime (setting H, Phase 4) and its accuracy cost.

Per model (released teacher checkpoint, or a trained student run folder):

1. Build the PyTorch model on CPU (eval) and fix the decoder's NMF starting bases to the tensor the evaluation
   draws after ``seed_all(0)``. ``evaluate.predict`` re-seeds 0 before every patch, so single-pass evaluation
   always uses this one tensor; with it stored as a buffer the exported graph has no random op and reproduces
   single-pass evaluation. Checked per model: fixed-bases output == random-init output after ``seed_all(0)``.
2. Export ONNX FP32 (opset 13, input 1×11×240×240).
3. Static INT8 quantization (ONNX Runtime, QDQ): Conv weights int8 per output channel, activations uint8,
   MinMax calibration on ``--calib`` *training* patches without augmentation. Only Conv nodes are quantized;
   the NMF matrix products / divisions stay FP32 (multiplicative updates are not robust to 8-bit rounding).
4. Single-pass accuracy of PyTorch FP32 (CPU), ONNX FP32 and ONNX INT8 on val and test, labelled pixels only,
   with the same bilinear upsampling + argmax as ``evaluate.predict``.
5. File sizes: PyTorch state dict, ONNX FP32, ONNX INT8.

Latency (``--latency RUN_DIR ...``): ONNX Runtime CPU, batch 1, 50 warm-up + 200 timed runs × 3 repeats,
median and p90, for each ``--cpu_threads``. Run it only when the laptop is otherwise idle (no training jobs).

Examples (env-orig, repo root):
    python -m src.debriskd.quantize --cache ~/mados_data/cache --teacher checkpoints/teacher/5/marinext_5.pth \
        --student runs/p3-F2-w0.25-s0-20261021-0312 --subset 20          # smoke test
    python -m src.debriskd.quantize --latency runs/p4-H-A-ckpt5-<ts> runs/p4-H-F2-w0.25-s0-<ts>
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
import yaml
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)

from .benchmark import patch_nmf_for_cpu, summarise
from .data import MADOSCached
from .metrics import confusion, scores_from_confusion
from .official import (CLASS_NAMES, DEBRIS, IGNORE_INDEX, NUM_CLASSES, MariNext, load_official_checkpoint,
                       seed_all)
from .runlog import env_info, make_run_id, run_dir, write_json
from .students import build_student, count_params

CPU = torch.device("cpu")
SHAPE = (1, 11, 240, 240)


def build_model(kind: str, src: str, width: float = 1.0) -> torch.nn.Module:
    """Teacher from an official checkpoint, or a student from ``<run>/ckpt/best_ema.pth`` (CPU, eval)."""
    if kind == "teacher":
        m = MariNext(11, NUM_CLASSES)
        load_official_checkpoint(m, src, CPU)
    else:
        m = build_student(width)
        m.load_state_dict(torch.load(src, map_location=CPU))
    return m.eval()


def fix_nmf_bases(model: torch.nn.Module, seed: int = 0) -> None:
    """Store the NMF bases drawn after ``seed_all(seed)`` (B = 1, CPU generator) as a buffer; no random op left."""
    head = model.decode_head
    ham = head.hamburger.ham
    D = head.ham_channels // ham.S
    torch.manual_seed(seed)
    bases = F.normalize(torch.rand((ham.S, D, ham.R)), dim=1)   # == NMF2D._build_bases(1, S, D, R) draw
    ham.rand_init = False
    ham.register_buffer("bases", bases)


def predict_from_logits(logits: torch.Tensor, out_hw) -> np.ndarray:
    """Same post-processing as ``evaluate.predict`` (single pass)."""
    logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
    return torch.softmax(logits, dim=1).argmax(1).numpy()


class Calib(CalibrationDataReader):
    """Feeds ``n`` training patches (fixed random choice, seed 0, no augmentation) to the calibrator.

    ``set_range`` lets ONNX Runtime calibrate in chunks (``CalibStridedMinMax``), so histogram calibrators do
    not hold every intermediate activation of all patches in memory at once.
    """

    def __init__(self, cache: str, n: int):
        ds = MADOSCached(cache, "train")
        ds.mode = "val"                                         # official __getitem__ augments only in 'train'
        idx = np.sort(np.random.RandomState(0).choice(len(ds), size=min(n, len(ds)), replace=False))
        self.data = [ds[i][0][None].astype(np.float32) for i in idx]
        self.set_range(0, len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def set_range(self, start_index: int, end_index: int) -> None:
        self.items = iter(self.data[start_index:end_index])

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        x = next(self.items, None)
        return None if x is None else {"image": x}


CALIB = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
         "entropy": CalibrationMethod.Entropy}


def session(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def evaluate(model: torch.nn.Module, sessions: Dict[str, ort.InferenceSession], ds) -> Dict[str, np.ndarray]:
    """Confusion matrices for PyTorch FP32 (CPU) and each ONNX session over one split."""
    cms = {k: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64) for k in ["torch_fp32", *sessions]}
    for i in range(len(ds)):
        image, target = ds[i]
        x = image[None].astype(np.float32)
        hw = tuple(target.shape[-2:])
        with torch.no_grad():
            preds = {"torch_fp32": predict_from_logits(model(torch.from_numpy(x)), hw)}
        for k, s in sessions.items():
            preds[k] = predict_from_logits(torch.from_numpy(s.run(None, {"image": x})[0]), hw)
        for k, p in preds.items():
            cms[k] += confusion(p, target[None], NUM_CLASSES, IGNORE_INDEX)
    return cms


def quantize_one(kind: str, src: str, width: float, tag: str, args) -> str:
    rid = make_run_id("p4", "H", tag)
    d = run_dir(rid)
    model = build_model(kind, src, width)
    sd = io.BytesIO()
    torch.save(model.state_dict(), sd)

    # Fixed bases reproduce the seeded random-init forward exactly (checked on 3 val patches).
    val = MADOSCached(args.cache, "val")
    xs = [torch.from_numpy(val[i][0][None].astype(np.float32)) for i in range(3)]
    with torch.no_grad():
        ref = []
        for x in xs:
            seed_all(0)
            ref.append(model(x))
        fix_nmf_bases(model)
        fixed = [model(x) for x in xs]
    same = all(torch.equal(a, b) for a, b in zip(ref, fixed))
    assert same, "fixed NMF bases do not reproduce the seed-0 forward"

    fp32, int8 = d / "model_fp32.onnx", d / "model_int8.onnx"
    torch.onnx.export(model, xs[0], str(fp32), opset_version=13, input_names=["image"], output_names=["logits"],
                      do_constant_folding=True)
    t0 = time.time()
    # ORT 1.19 sets a [0, 1] range on every Softmax output, which fails unless that tensor was calibrated.
    # So Softmax is calibrated but all Softmax nodes are excluded: only Conv nodes end up quantized.
    nodes = onnx.load(str(fp32)).graph.node
    exclude = [n.name for n in nodes if n.op_type == "Softmax"]
    convs = [n.name for n in nodes if n.op_type == "Conv"]
    if args.keep_ends_fp32:                                     # first (11-band input) and last (classifier) conv
        exclude += [convs[0], convs[-1]]
    reader = Calib(args.cache, args.calib)
    quantize_static(str(fp32), str(int8), reader, quant_format=QuantFormat.QDQ,
                    op_types_to_quantize=["Conv", "Softmax"], nodes_to_exclude=exclude,
                    per_channel=True, reduce_range=args.reduce_range,
                    activation_type=QuantType.QInt8 if args.act_type == "int8" else QuantType.QUInt8,
                    weight_type=QuantType.QInt8, calibrate_method=CALIB[args.calib_method],
                    extra_options={"CalibStridedMinMax": 10 if len(reader) % 10 == 0 else len(reader),
                                   "CalibPercentile": args.percentile,
                                   "ActivationSymmetric": args.act_type == "int8"})
    calib_s = time.time() - t0
    sessions = {"onnx_fp32": session(fp32, args.threads), "onnx_int8": session(int8, args.threads)}
    with torch.no_grad():
        diff = max(float((torch.from_numpy(sessions["onnx_fp32"].run(None, {"image": x.numpy()})[0]) - f).abs().max())
                   for x, f in zip(xs, fixed))

    results = {}
    for split in args.splits:
        ds = MADOSCached(args.cache, split)
        idx = range(min(args.subset, len(ds))) if args.subset else range(len(ds))
        ds_sub = [ds[i] for i in idx] if args.subset else ds
        cms = evaluate(model, sessions, ds_sub)
        results[split] = {k: scores_from_confusion(cm, CLASS_NAMES) for k, cm in cms.items()}
        results[split]["n_patches"] = len(ds_sub)
        for k in cms:
            r = results[split][k]; db = r["per_class"][CLASS_NAMES[DEBRIS]]
            print(f"{rid} {split:4s} {k:10s} debris F1 {100*db['f1']:.1f} R {100*db['recall']:.1f} "
                  f"P {100*db['precision']:.1f} | macroF1 {100*r['macro_f1']:.1f} mIoU {100*r['mIoU']:.1f}", flush=True)

    cfg = {**{k: v for k, v in vars(args).items() if k not in ("teacher", "student", "latency")},
           "run_id": rid, "kind": kind, "source": src, "width": width, "tag": tag,
           "quantization": {"variant": args.variant, "format": "QDQ", "op_types": ["Conv"], "per_channel": True,
                            "activations": args.act_type, "weights": "int8", "reduce_range": args.reduce_range,
                            "first_last_conv_fp32": args.keep_ends_fp32,
                            "calibration": f"{args.calib_method}"
                                           + (f" {args.percentile}" if args.calib_method == "percentile" else "")
                                           + f", {args.calib} train patches (seed 0, no aug)",
                            "opset": 13, "nmf_bases": "fixed = seed_all(0) draw"}}
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(d / "metrics.json", {
        "run_id": rid, "kind": kind, "source": src, "width": width, "params": count_params(model),
        "sizes_bytes": {"torch_state_dict": sd.getbuffer().nbytes, "onnx_fp32": fp32.stat().st_size,
                        "onnx_int8": int8.stat().st_size},
        "checks": {"fixed_bases_equal_seeded_forward": same, "onnx_fp32_max_abs_diff_vs_torch": diff},
        "calibration_s": calib_s, "results": results})
    write_json(d / "env.json", {**env_info(), "onnxruntime": ort.__version__})
    return rid


def latency(run_dirs: List[str], threads: List[int], warmup: int, timed: int, repeats: int) -> None:
    """ONNX Runtime CPU latency of model_fp32.onnx / model_int8.onnx; adds ``latency`` to each metrics.json."""
    import json
    x = np.random.RandomState(0).randn(*SHAPE).astype(np.float32)
    for rd in map(Path, run_dirs):
        m = json.loads((rd / "metrics.json").read_text())
        lat = {}
        for name in ("onnx_fp32", "onnx_int8"):
            for n in threads:
                s = session(rd / f"model_{name[5:]}.onnx", n)
                runs = []
                for _ in range(repeats):
                    for _ in range(warmup):
                        s.run(None, {"image": x})
                    ts = []
                    for _ in range(timed):
                        t0 = time.perf_counter(); s.run(None, {"image": x}); ts.append((time.perf_counter() - t0) * 1000)
                    runs.append(ts)
                lat[f"{name}_cpu_{n}t"] = summarise(runs)
                print(rd.name, name, f"{n}t", f"{lat[f'{name}_cpu_{n}t']['median_ms']:.2f} ms", flush=True)
        m["latency"] = {"protocol": {"batch": 1, "input": list(SHAPE), "warmup": warmup, "timed": timed,
                                     "repeats": repeats, "engine": f"onnxruntime {ort.__version__} CPU"}, **lat}
        write_json(rd / "metrics.json", m)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=str(Path.home() / "mados_data" / "cache"))
    p.add_argument("--teacher", nargs="*", default=[], help="official teacher checkpoints (.pth)")
    p.add_argument("--student", nargs="*", default=[], help="student run folders (runs/<run_id>)")
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    p.add_argument("--calib", type=int, default=200, help="number of training patches for calibration")
    p.add_argument("--threads", type=int, default=4, help="CPU threads for the accuracy evaluation")
    p.add_argument("--subset", type=int, default=0, help="smoke test: evaluate only N patches per split")
    p.add_argument("--epochs", type=int, default=None, help="unused; accepted for CLI uniformity")
    p.add_argument("--variant", default="plain", help="name of the quantization setting (appended to the run tag)")
    p.add_argument("--calib_method", default="minmax", choices=sorted(CALIB))
    p.add_argument("--percentile", type=float, default=99.999, help="for --calib_method percentile")
    p.add_argument("--act_type", default="uint8", choices=["uint8", "int8"], help="int8 = symmetric activations")
    p.add_argument("--reduce_range", action="store_true", help="7-bit weights (avoids AVX2 U8S8 saturation)")
    p.add_argument("--keep_ends_fp32", action="store_true", help="leave the first and last Conv in FP32")
    p.add_argument("--latency", nargs="*", default=None, help="measure latency of these p4-H run folders only")
    p.add_argument("--cpu_threads", nargs="+", type=int, default=[8, 1])
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--timed", type=int, default=200)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    patch_nmf_for_cpu()                      # seeded reference forward on CPU (bases follow the input device)
    torch.set_num_threads(args.threads)
    if args.latency is not None:
        latency(args.latency, args.cpu_threads, args.warmup, args.timed, args.repeats)
        return
    sfx = "" if args.variant == "plain" else f"-q{args.variant}"
    for ck in args.teacher:
        print("wrote", quantize_one("teacher", ck, 1.0, f"A-ckpt{Path(ck).parent.name}{sfx}", args))
    for rd in map(Path, args.student):
        c = yaml.safe_load((rd / "config.yaml").read_text())
        tag = f"{c['setting']}-w{c['width']}-s{c['seed']}{sfx}"
        print("wrote", quantize_one("student", str(rd / "ckpt" / "best_ema.pth"), float(c["width"]), tag, args))


if __name__ == "__main__":
    main()
