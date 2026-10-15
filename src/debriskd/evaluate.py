"""Evaluate MariNeXt checkpoints on labelled pixels only, single-pass and with the official TTA.

Mirrors the inference loop of ``third_party/mados/marinext/evaluation.py`` exactly:
batch 1, ``seed_all(0)`` before every forward (the LightHam decoder draws random NMF bases on every
forward, even in eval mode), bilinear upsampling to 240×240, argmax; TTA = official ``utils/test_time_aug.TTA``
(8 views, per-pixel majority vote of hard labels). Metrics come from our confusion matrix (``metrics.py``).

Each checkpoint gets its own run folder ``runs/<run_id>/`` with ``config.yaml``, ``metrics.json``, ``env.json``.

Example (env-orig, repo root):
    python -m src.debriskd.evaluate --path ~/mados_data/MADOS --ckpt checkpoints/teacher/1/marinext_1.pth \
        --splits val test --modes single tta --phase p1 --setting A --tag ckpt1-envorig
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .metrics import confusion, scores_from_confusion
from .official import (CLASS_NAMES, DEBRIS, IGNORE_INDEX, MADOS, NUM_CLASSES, TTA, MariNext,
                       load_official_checkpoint, seed_all, split_dir)
from .runlog import disable_tf32, env_info, make_run_id, run_dir, write_json


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def predict(model: torch.nn.Module, image: torch.Tensor, out_hw, tta: bool) -> torch.Tensor:
    """Hard prediction [1, H, W] for one patch, exactly as in the official evaluation loop."""
    if tta:
        image = TTA(image)
    seed_all(0)
    logits = model(image)
    logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)  # == F.upsample default
    pred = torch.softmax(logits, dim=1).argmax(1)
    if tta:
        pred = TTA(pred, reverse_aggregation=True)
    return pred


def evaluate_split(models: Dict[str, torch.nn.Module], dataset, modes: List[str], device) -> Dict[str, Dict[str, np.ndarray]]:
    """Confusion matrices ``cms[ckpt][mode]`` for every model and mode over one dataset."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    cms = {k: {m: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64) for m in modes} for k in models}
    for image, target in tqdm(loader, desc="eval", mininterval=30):
        target_np = target.numpy()
        for name, model in models.items():
            for mode in modes:
                pred = predict(model, image.to(device), tuple(target.shape[-2:]), tta=(mode == "tta"))
                cms[name][mode] += confusion(pred.cpu().numpy(), target_np, NUM_CLASSES, IGNORE_INDEX)
    return cms


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True, help="MADOS root (contains Scene_* and splits/)")
    p.add_argument("--ckpt", nargs="+", required=True, help="checkpoint .pth files (official format)")
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    p.add_argument("--modes", nargs="+", default=["single", "tta"], choices=["single", "tta"])
    p.add_argument("--phase", default="p1")
    p.add_argument("--setting", default="A")
    p.add_argument("--tag", nargs="+", default=None, help="one run tag per checkpoint (default: file stem)")
    p.add_argument("--subset", type=int, default=0, help="evaluate only the first N patches of each split")
    p.add_argument("--epochs", type=int, default=None, help="unused; accepted for CLI uniformity")
    p.add_argument("--tf32", action="store_true",
                   help="leave TF32 at the torch default (only to reproduce official evaluation.py bit-for-bit)")
    args = p.parse_args()

    if not args.tf32:
        disable_tf32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tags = args.tag or [Path(c).stem for c in args.ckpt]
    assert len(tags) == len(args.ckpt), "--tag needs one entry per --ckpt"

    models: Dict[str, torch.nn.Module] = {}
    for ck in args.ckpt:
        m = MariNext(11, NUM_CLASSES).to(device)
        load_official_checkpoint(m, ck, device)
        m.eval()
        models[ck] = m

    results: Dict[str, Dict] = {ck: {} for ck in args.ckpt}
    timing: Dict[str, float] = {}
    for split in args.splits:
        t0 = time.time()
        ds = MADOS(args.path, split_dir(args.path), split)
        timing[f"load_{split}_s"] = time.time() - t0
        if args.subset:
            ds = Subset(ds, range(min(args.subset, len(ds))))
        t0 = time.time()
        cms = evaluate_split(models, ds, args.modes, device)
        timing[f"eval_{split}_s"] = time.time() - t0
        for ck in args.ckpt:
            results[ck][split] = {m: scores_from_confusion(cms[ck][m], CLASS_NAMES) for m in args.modes}
            results[ck][split]["n_patches"] = len(ds)
        del ds

    env = env_info()
    for ck, tag in zip(args.ckpt, tags):
        rid = make_run_id(args.phase, args.setting, tag)
        d = run_dir(rid)
        cfg = {**vars(args), "ckpt": ck, "tag": tag, "run_id": rid, "kind": "evaluate",
               "ckpt_sha256": sha256(ck), "debris_index": DEBRIS, "ignore_index": IGNORE_INDEX,
               "class_names": list(CLASS_NAMES)}
        (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        write_json(d / "metrics.json", {"run_id": rid, "ckpt": ck, "results": results[ck], "timing": timing})
        write_json(d / "env.json", env)
        for split, r in results[ck].items():
            for mode in args.modes:
                dbr = r[mode]["per_class"][CLASS_NAMES[DEBRIS]]
                print(f"{rid} {split:5s} {mode:6s} debris F1 {100*dbr['f1']:.1f} R {100*dbr['recall']:.1f} "
                      f"IoU {100*dbr['iou']:.1f} | mIoU {100*r[mode]['mIoU']:.1f} macroF1 {100*r[mode]['macro_f1']:.1f}")


if __name__ == "__main__":
    main()
