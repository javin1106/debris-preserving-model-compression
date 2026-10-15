"""Per-class segmentation metrics on labelled pixels only (ignore index excluded before counting)."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def confusion(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int = -1) -> np.ndarray:
    """Confusion matrix ``cm[true, pred]`` over pixels whose target != ignore_index."""
    pred = np.asarray(pred).ravel()
    target = np.asarray(target).ravel()
    keep = target != ignore_index
    t, p = target[keep].astype(np.int64), pred[keep].astype(np.int64)
    if t.size and (t.min() < 0 or t.max() >= num_classes or p.min() < 0 or p.max() >= num_classes):
        raise ValueError("class index out of range")
    return np.bincount(t * num_classes + p, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise a / b with 0 where b == 0 (sklearn's zero_division behaviour)."""
    out = np.zeros_like(a, dtype=np.float64)
    np.divide(a, b, out=out, where=b > 0)
    return out


def scores_from_confusion(cm: np.ndarray, class_names: Sequence[str]) -> Dict:
    """Per-class precision/recall/F1/IoU, plus mIoU, macro-F1, macro-P/R, overall accuracy.

    Macro averages are taken over classes present in targets or predictions (as sklearn does);
    the per-class dict also records ``support`` (labelled pixels of that class).
    """
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
    f1 = _div(2 * tp, 2 * tp + fp + fn)
    iou = _div(tp, tp + fp + fn)
    present = (cm.sum(0) + cm.sum(1)) > 0
    per_class: Dict[str, Dict] = {}
    for i, n in enumerate(class_names):
        per_class[n] = {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                        "iou": float(iou[i]), "support": int(cm[i].sum())}
    return {
        "per_class": per_class,
        "mIoU": float(iou[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "macro_precision": float(prec[present].mean()),
        "macro_recall": float(rec[present].mean()),
        "overall_acc": float(tp.sum() / max(cm.sum(), 1)),
        "n_labelled_px": int(cm.sum()),
        "confusion": cm.astype(int).tolist(),
    }
