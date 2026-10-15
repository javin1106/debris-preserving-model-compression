"""Student builder: shapes, parameter counts, width 1.0 == teacher."""
import pytest
import torch

from src.debriskd.official import MariNext
from src.debriskd.students import build_student, count_params

TARGETS = {1.0: 4_238_063, 0.5: 1_130_751, 0.25: 318_311, 0.125: 97_395}
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="NMF decoder calls .cuda() unconditionally")


def test_width1_is_teacher():
    t, s = MariNext(11, 15), build_student(1.0)
    assert count_params(t) == count_params(s) == TARGETS[1.0]
    ts, ss = t.state_dict(), s.state_dict()
    assert list(ts.keys()) == list(ss.keys())
    assert all(ts[k].shape == ss[k].shape for k in ts)


@pytest.mark.parametrize("width", [0.5, 0.25, 0.125])
def test_param_counts_within_2pct(width):
    n = count_params(build_student(width))
    assert abs(n - TARGETS[width]) / TARGETS[width] < 0.02, n


@cuda
@pytest.mark.parametrize("width", [1.0, 0.5, 0.25, 0.125])
def test_forward_shape_matches_teacher(width):
    torch.manual_seed(0)
    x = torch.randn(1, 11, 240, 240).cuda()
    with torch.no_grad():
        ref = MariNext(11, 15).cuda().eval()(x)
        out = build_student(width).cuda().eval()(x)
    assert out.shape == ref.shape == (1, 15, 60, 60)
    assert torch.isfinite(out).all()


def test_groupnorm_fallback_for_odd_width():
    m = build_student(3 / 16)                     # decoder width 48 → gcd(32, 48) = 16 groups
    gns = [mod for mod in m.modules() if isinstance(mod, torch.nn.GroupNorm)]
    assert gns and all(g.num_channels % g.num_groups == 0 for g in gns)
