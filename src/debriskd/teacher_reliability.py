"""How reliable is the KD teacher where it is confident? (supports the choice of τ and the "evidence, not labels" rule)

For every patch the teacher runs exactly as in single-pass evaluation (batch 1, ``seed_all(0)`` before the forward,
bilinear upsampling, softmax at T = 1, no augmentation). Counts are accumulated per split:

* labelled pixels, binned by teacher confidence c = max softmax: number of pixels and number the teacher gets right
  (a reliability table);
* labelled pixels with c ≥ τ, per true class: how many there are and how many the teacher gets right
  (coverage and accuracy of the pixels the unlabelled-pixel KD term would use, if they had been unlabelled);
* unlabelled (ignore-index) pixels: total, c ≥ τ, and the teacher's class histogram over c ≥ τ.

Writes ``runs/<run_id>/{config.yaml,metrics.json,env.json}``. The teacher was trained on the train split, so train
accuracies are optimistic; val is the honest estimate.

Example (env-orig, repo root):
    python -m src.debriskd.teacher_reliability --cache ~/mados_data/cache \
        --teacher checkpoints/teacher/5/marinext_5.pth --splits train val
"""
from __future__ import annotations

import argparse
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data import MADOSCached
from .official import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, MariNext, load_official_checkpoint, seed_all
from .runlog import disable_tf32, env_info, make_run_id, run_dir, write_json

BINS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0 + 1e-9)


def empty_counts(n_classes: int = NUM_CLASSES, bins: Sequence[float] = BINS) -> Dict[str, np.ndarray]:
    """Zeroed accumulators for :func:`reliability_counts`."""
    nb = len(bins) - 1
    return {"bin_n": np.zeros(nb, np.int64), "bin_correct": np.zeros(nb, np.int64),
            "conf_lab_n": np.zeros(n_classes, np.int64), "conf_lab_correct": np.zeros(n_classes, np.int64),
            "lab_n": np.zeros(n_classes, np.int64),
            "pred_conf_lab_n": np.zeros(n_classes, np.int64), "pred_conf_lab_correct": np.zeros(n_classes, np.int64),
            "ign_n": np.zeros(1, np.int64), "ign_conf_n": np.zeros(1, np.int64),
            "ign_conf_hist": np.zeros(n_classes, np.int64)}


def reliability_counts(prob: np.ndarray, target: np.ndarray, tau: float, acc: Dict[str, np.ndarray],
                       bins: Sequence[float] = BINS, ignore_index: int = IGNORE_INDEX) -> None:
    """Add one patch's counts to ``acc`` in place.

    prob: [C, H, W] teacher softmax at T = 1; target: [H, W] labels with ``ignore_index`` for unlabelled pixels.
    Labelled-pixel counts never include ignored pixels and vice versa.
    """
    n_classes = prob.shape[0]
    conf, pred = prob.max(0), prob.argmax(0)
    lab = target != ignore_index
    c_l, p_l, t_l = conf[lab], pred[lab], target[lab]
    ok = p_l == t_l
    b = np.clip(np.digitize(c_l, bins) - 1, 0, len(bins) - 2)
    acc["bin_n"] += np.bincount(b, minlength=len(bins) - 1)
    acc["bin_correct"] += np.bincount(b[ok], minlength=len(bins) - 1)
    hi = c_l >= tau
    acc["lab_n"] += np.bincount(t_l, minlength=n_classes)
    acc["conf_lab_n"] += np.bincount(t_l[hi], minlength=n_classes)
    acc["conf_lab_correct"] += np.bincount(t_l[hi & ok], minlength=n_classes)
    acc["pred_conf_lab_n"] += np.bincount(p_l[hi], minlength=n_classes)
    acc["pred_conf_lab_correct"] += np.bincount(p_l[hi & ok], minlength=n_classes)
    c_u, p_u = conf[~lab], pred[~lab]
    acc["ign_n"] += c_u.size
    acc["ign_conf_n"] += int((c_u >= tau).sum())
    acc["ign_conf_hist"] += np.bincount(p_u[c_u >= tau], minlength=n_classes)


def summarise(acc: Dict[str, np.ndarray], bins: Sequence[float] = BINS) -> Dict:
    """JSON-ready summary of the accumulated counts (rates as fractions; NaN-free: empty → None)."""
    def rate(a, b):
        return [None if d == 0 else float(n) / float(d) for n, d in zip(a, b)]

    n = len(acc["lab_n"])
    names = list(CLASS_NAMES) if n == NUM_CLASSES else [str(i) for i in range(n)]
    labels = [f"[{bins[i]:.2f}, {min(bins[i + 1], 1.0):.2f}{']' if i == len(bins) - 2 else ')'}"
              for i in range(len(bins) - 1)]
    tot = int(acc["bin_n"].sum())
    return {
        "labelled_px": tot,
        "labelled_accuracy": float(acc["bin_correct"].sum()) / tot if tot else None,
        "reliability": [{"conf_bin": l, "n": int(n), "share": float(n) / tot if tot else None, "accuracy": a}
                        for l, n, a in zip(labels, acc["bin_n"], rate(acc["bin_correct"], acc["bin_n"]))],
        "per_class_true": {nm: {"labelled_px": int(acc["lab_n"][i]), "confident_px": int(acc["conf_lab_n"][i]),
                                "coverage": rate([acc["conf_lab_n"][i]], [acc["lab_n"][i]])[0],
                                "accuracy_when_confident": rate([acc["conf_lab_correct"][i]],
                                                                [acc["conf_lab_n"][i]])[0]}
                           for i, nm in enumerate(names)},
        "per_class_predicted": {nm: {"confident_predictions": int(acc["pred_conf_lab_n"][i]),
                                     "precision_when_confident": rate([acc["pred_conf_lab_correct"][i]],
                                                                      [acc["pred_conf_lab_n"][i]])[0]}
                                for i, nm in enumerate(names)},
        "unlabelled_px": int(acc["ign_n"][0]),
        "unlabelled_confident_share": float(acc["ign_conf_n"][0]) / acc["ign_n"][0] if acc["ign_n"][0] else None,
        "unlabelled_confident_hist": {nm: int(acc["ign_conf_hist"][i]) for i, nm in enumerate(names)},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True, help="verified .npy cache dir (src/debriskd/data.py)")
    p.add_argument("--teacher", default="checkpoints/teacher/5/marinext_5.pth", help="KD teacher (official format)")
    p.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    p.add_argument("--tau", type=float, default=0.8, help="confidence threshold used by the KD runs")
    p.add_argument("--subset", type=int, default=0, help="use only the first N patches of each split")
    p.add_argument("--epochs", type=int, default=None, help="unused; accepted for CLI uniformity")
    args = p.parse_args()

    disable_tf32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MariNext(11, NUM_CLASSES).to(device)
    load_official_checkpoint(model, args.teacher, device)
    model.eval()

    results = {}
    for split in args.splits:
        ds = MADOSCached(args.cache, split)
        ds.mode = "val"                                          # official __getitem__ augments only in 'train'
        if args.subset:
            ds = Subset(ds, range(min(args.subset, len(ds))))
        acc = empty_counts()
        with torch.no_grad():
            for image, target in tqdm(DataLoader(ds, batch_size=1, shuffle=False), desc=split, mininterval=30):
                seed_all(0)
                logits = F.interpolate(model(image.to(device)), size=tuple(target.shape[-2:]), mode="bilinear",
                                       align_corners=False)
                prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
                reliability_counts(prob, target[0].numpy().astype(np.int64), args.tau, acc)
        results[split] = {**summarise(acc), "n_patches": len(ds)}
        r = results[split]
        print(f"{split}: labelled acc {r['labelled_accuracy']:.4f}, unlabelled confident share "
              f"{r['unlabelled_confident_share']:.4f}")

    rid = make_run_id("p3", "teacherrel", "ckpt5" + (f"-sub{args.subset}" if args.subset else ""))
    d = run_dir(rid)
    (d / "config.yaml").write_text(yaml.safe_dump({**vars(args), "run_id": rid, "kind": "teacher_reliability",
                                                   "bins": list(BINS)}, sort_keys=False), encoding="utf-8")
    write_json(d / "metrics.json", {"run_id": rid, "tau": args.tau, "results": results})
    write_json(d / "env.json", env_info())
    print("wrote", d)


if __name__ == "__main__":
    main()
