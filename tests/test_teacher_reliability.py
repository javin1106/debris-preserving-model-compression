"""Teacher-reliability counts: labelled and ignored pixels are kept apart; bins and τ split as documented."""
import numpy as np

from src.debriskd.teacher_reliability import empty_counts, reliability_counts, summarise


def _prob(classes, confs, n_classes=3):
    """[C, 1, W] softmax-like array whose argmax is ``classes`` with max value ``confs``."""
    w = len(classes)
    p = np.zeros((n_classes, 1, w))
    for j, (c, v) in enumerate(zip(classes, confs)):
        p[:, 0, j] = (1 - v) / (n_classes - 1)
        p[c, 0, j] = v
    return p


def test_counts_split_labelled_and_ignored():
    prob = _prob([0, 1, 2, 0, 1], [0.95, 0.85, 0.55, 0.9, 0.99])
    target = np.array([[0, 2, 2, -1, -1]])
    acc = empty_counts(3)
    reliability_counts(prob, target, 0.8, acc, ignore_index=-1)
    assert acc["bin_n"].sum() == 3 and acc["bin_correct"].sum() == 2       # px 0, 2 right; px 1 wrong
    assert acc["lab_n"].tolist() == [1, 0, 2]
    assert acc["conf_lab_n"].tolist() == [1, 0, 1]                         # px 2 (0.55) is below τ
    assert acc["conf_lab_correct"].tolist() == [1, 0, 0]
    assert acc["pred_conf_lab_n"].tolist() == [1, 1, 0]
    assert acc["pred_conf_lab_correct"].tolist() == [1, 0, 0]
    assert acc["ign_n"][0] == 2 and acc["ign_conf_n"][0] == 2
    assert acc["ign_conf_hist"].tolist() == [1, 1, 0]


def test_ignored_predictions_do_not_change_labelled_counts():
    target = np.array([[0, -1, 1, -1]])
    a, b = empty_counts(3), empty_counts(3)
    reliability_counts(_prob([0, 0, 1, 0], [0.9, 0.9, 0.7, 0.9]), target, 0.8, a, ignore_index=-1)
    reliability_counts(_prob([0, 2, 1, 1], [0.9, 0.3, 0.7, 0.99]), target, 0.8, b, ignore_index=-1)
    for k in ("bin_n", "bin_correct", "lab_n", "conf_lab_n", "conf_lab_correct"):
        assert np.array_equal(a[k], b[k])


def test_summary_rates():
    acc = empty_counts(3)
    reliability_counts(_prob([0, 0], [0.97, 0.97]), np.array([[0, 1]]), 0.8, acc, ignore_index=-1)
    s = summarise(acc)
    assert s["labelled_accuracy"] == 0.5
    row = [r for r in s["reliability"] if r["n"]][0]
    assert row["conf_bin"].startswith("[0.95") and row["accuracy"] == 0.5
