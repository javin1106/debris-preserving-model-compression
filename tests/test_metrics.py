"""Metric tests: ignore pixels never count; F1 = 2·IoU/(1+IoU); agreement with sklearn on labelled pixels."""
import numpy as np
import pytest

from src.debriskd.metrics import confusion, scores_from_confusion

NAMES = ["a", "b", "c"]


def test_ignore_pixels_never_counted():
    target = np.array([[0, 1, -1], [2, -1, 1]])
    pred = np.array([[0, 1, 2], [2, 0, 1]])
    base = confusion(pred, target, 3)
    assert base.sum() == 4                       # only the 4 labelled pixels
    # Changing predictions on ignored pixels must not change anything.
    pred2 = pred.copy()
    pred2[target == -1] = (pred2[target == -1] + 1) % 3
    assert np.array_equal(confusion(pred2, target, 3), base)
    s = scores_from_confusion(base, NAMES)
    assert s["overall_acc"] == 1.0               # all labelled pixels correct → no TP/FP/FN from ignored ones
    for n in NAMES:
        assert s["per_class"][n]["f1"] == 1.0


def test_all_ignored_gives_empty_confusion():
    cm = confusion(np.zeros((4, 4), int), -np.ones((4, 4), int), 3)
    assert cm.sum() == 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_f1_iou_identity_and_sklearn(seed):
    from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

    rng = np.random.RandomState(seed)
    target = rng.randint(-1, 3, size=(50, 50))
    pred = rng.randint(0, 3, size=(50, 50))
    s = scores_from_confusion(confusion(pred, target, 3), NAMES)
    for n in NAMES:
        f1, iou = s["per_class"][n]["f1"], s["per_class"][n]["iou"]
        assert f1 == pytest.approx(2 * iou / (1 + iou), abs=1e-12)
    keep = target != -1
    t, p = target[keep], pred[keep]
    assert s["macro_f1"] == pytest.approx(f1_score(t, p, average="macro"), abs=1e-12)
    assert s["mIoU"] == pytest.approx(jaccard_score(t, p, average="macro"), abs=1e-12)
    np.testing.assert_allclose([s["per_class"][n]["precision"] for n in NAMES], precision_score(t, p, average=None))
    np.testing.assert_allclose([s["per_class"][n]["recall"] for n in NAMES], recall_score(t, p, average=None))


def test_out_of_range_rejected():
    with pytest.raises(ValueError):
        confusion(np.array([3]), np.array([0]), 3)
