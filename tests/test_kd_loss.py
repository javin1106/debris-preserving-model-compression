"""KD loss tests (Phase 3.2): masking, confidence weights, reduction to L_sup, gradient flow."""
import pytest
import torch
import torch.nn.functional as F

from src.debriskd.kd_loss import KDConfig, kd_terms, lam_U_at, total_loss
from src.debriskd.official import DEBRIS, IGNORE_INDEX

C = 15


def batch(seed=0, B=2, H=8, W=8):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(B, C, H, W, generator=g) * 3
    s = torch.randn(B, C, H, W, generator=g, requires_grad=True)
    y = torch.randint(0, C, (B, H, W), generator=g)
    y[:, :, : W // 2] = IGNORE_INDEX            # left half unlabelled
    return s, t, y


def test_constants_from_dataset():
    assert DEBRIS == 0 and IGNORE_INDEX == -1


def test_identical_logits_give_zero_kd():
    _, t, y = batch()
    k = kd_terms(t.clone().requires_grad_(True), t, y, KDConfig(tau=0.0))
    assert k["kd_lab"].abs() < 1e-6 and k["kd_ign"].abs() < 1e-6


def test_labelled_term_ignores_ignored_pixels():
    s, t, y = batch()
    cfg = KDConfig(tau=0.0)
    base = kd_terms(s, t, y, cfg)["kd_lab"]
    s2, t2 = s.detach().clone(), t.clone()
    ign = (y == IGNORE_INDEX).unsqueeze(1).expand_as(s2)
    s2[ign] = torch.randn(int(ign.sum())); t2[ign] = torch.randn(int(ign.sum()))
    assert torch.allclose(kd_terms(s2, t2, y, cfg)["kd_lab"], base)


def test_ignored_term_ignores_labelled_pixels():
    s, t, y = batch()
    cfg = KDConfig(tau=0.0)
    base = kd_terms(s, t, y, cfg)["kd_ign"]
    s2, t2 = s.detach().clone(), t.clone()
    lab = (y != IGNORE_INDEX).unsqueeze(1).expand_as(s2)
    s2[lab] = torch.randn(int(lab.sum())); t2[lab] = torch.randn(int(lab.sum()))
    assert torch.allclose(kd_terms(s2, t2, y, cfg)["kd_ign"], base)


def test_sup_loss_ignores_ignored_pixels():
    s, t, y = batch()
    ce = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    s2 = s.detach().clone()
    s2[(y == IGNORE_INDEX).unsqueeze(1).expand_as(s2)] = 1e3
    assert torch.allclose(ce(s.detach(), y), ce(s2, y))


@pytest.mark.parametrize("tau", [0.6, 0.8, 0.9])
def test_low_confidence_pixels_contribute_nothing(tau):
    s, t, y = batch()
    cfg = KDConfig(tau=tau)
    k = kd_terms(s, t, y, cfg)
    assert (k["w"] >= 0).all() and (k["w"] <= 1).all()
    conf = F.softmax(t, 1).amax(1)
    assert (k["w"][conf <= tau] == 0).all()
    # Changing student logits only where conf <= tau leaves kd_ign unchanged.
    s2 = s.detach().clone()
    low = ((conf <= tau) & (y == IGNORE_INDEX)).unsqueeze(1).expand_as(s2)
    s2[low] = torch.randn(int(low.sum()))
    assert torch.allclose(kd_terms(s2, t, y, cfg)["kd_ign"], k["kd_ign"])


def test_debris_weight_applied_and_clipped():
    s, t, y = batch()
    t[:, DEBRIS] += 20                                   # teacher says debris everywhere, confidently
    k1 = kd_terms(s, t, y, KDConfig(tau=0.5, debris_w=1))["kd_ign"]
    k3 = kd_terms(s, t, y, KDConfig(tau=0.5, debris_w=3, debris_w_max=4))["kd_ign"]
    k9 = kd_terms(s, t, y, KDConfig(tau=0.5, debris_w=9, debris_w_max=4))["kd_ign"]
    assert torch.allclose(k3, 3 * k1) and torch.allclose(k9, 4 * k1)


def test_zero_lambdas_reproduce_sup():
    s, t, y = batch()
    L_sup = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(s, y)
    out = total_loss(L_sup, s, t, y, KDConfig(lam_L=0, lam_U=0), epoch=10)
    assert torch.equal(out["loss"], L_sup)


def test_gradient_flows_to_student_only():
    torch.manual_seed(0)
    teacher, student = torch.nn.Conv2d(3, C, 1), torch.nn.Conv2d(3, C, 1)
    teacher.requires_grad_(False)
    x = torch.randn(2, 3, 8, 8)
    y = torch.randint(-1, C, (2, 8, 8))
    with torch.no_grad():
        t = teacher(x)
    s = student(x)
    L_sup = torch.nn.CrossEntropyLoss(ignore_index=-1)(s, y)
    total_loss(L_sup, s, t, y, KDConfig(tau=0.0), epoch=10)["loss"].backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())


def test_warmup():
    cfg = KDConfig(lam_U=1.0, warmup_epochs=5)
    assert lam_U_at(1, cfg) == 0.0 and lam_U_at(6, cfg) == 1.0 and 0 < lam_U_at(3, cfg) < 1


def test_no_eligible_pixels_is_zero_not_nan():
    s, t, y = batch()
    k = kd_terms(s, t, y, KDConfig(tau=0.999999))
    assert torch.isfinite(k["kd_ign"]) and k["kd_ign"] == 0


def test_class_weighted_kd():
    s, t, y = batch()
    ones = KDConfig(tau=0.0, class_weights=torch.ones(C))
    plain = KDConfig(tau=0.0)
    a, b = kd_terms(s, t, y, ones), kd_terms(s, t, y, plain)
    assert torch.allclose(a["kd_lab"], b["kd_lab"]) and torch.allclose(a["kd_ign"], b["kd_ign"])  # uniform = plain
    cw = torch.ones(C); cw[DEBRIS] = 10.0
    y2 = y.clone(); y2[y2 != IGNORE_INDEX] = DEBRIS                         # all labelled px are debris
    k = kd_terms(s, t, y2, KDConfig(tau=0.0, class_weights=cw))
    assert torch.allclose(k["kd_lab"], kd_terms(s, t, y2, plain)["kd_lab"])  # single class → same weighted mean
    # Ignored pixels still never enter the labelled term with class weights.
    s2 = s.detach().clone(); ign = (y2 == IGNORE_INDEX).unsqueeze(1).expand_as(s2)
    s2[ign] = torch.randn(int(ign.sum()))
    assert torch.allclose(kd_terms(s2, t, y2, KDConfig(tau=0.0, class_weights=cw))["kd_lab"], k["kd_lab"])
