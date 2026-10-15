"""Width-scaled MariNeXt students built from the official config and classes (Phase 2.1).

``build_student(width)`` loads ``marinext/configs/marinext.tiny.240x240.mados.py`` exactly as the official
``MariNext`` wrapper does, then scales only channel widths:

* encoder ``embed_dims`` (32, 64, 160, 256) × width, rounded, minimum 4 (and even — ``StemConv`` uses ``dim // 2``);
* decoder ``in_channels`` follow the encoder; ``channels`` and ``ham_channels`` = 256 × width (rounded, minimum 4).

Everything else (depths 3-3-5-2, MLP ratios 8-8-4-4, drop-path 0.1, 11 bands, 15 classes, NMF rank ``MD_R = 16``,
dropout 0.1, norm layers) is unchanged.

Divisibility: the decoder uses ``GroupNorm(num_groups=32)``. When 32 does not divide the decoder width, the group
count becomes ``gcd(32, channels)`` (only for widths such as 3/16; ½, ¼ and ⅛ keep 32 groups). Width 1.0 returns
exactly the teacher architecture (same modules, same parameter count, same state-dict keys).
"""
from __future__ import annotations

import math
import os

from torch import nn

from . import official  # noqa: F401  (puts the official code on sys.path)
from mmcv.cnn.utils import revert_sync_batchnorm  # noqa: E402
from mmcv.utils import Config  # noqa: E402
from mmseg.models import build_segmentor  # noqa: E402

CONFIG = os.path.join(str(official.MADOS_DIR), "marinext", "configs", "marinext.tiny.240x240.mados.py")
TEACHER_EMBED = (32, 64, 160, 256)
TEACHER_DECODER = 256
GN_GROUPS = 32


def _scale(c: int, width: float, even: bool = False) -> int:
    v = max(4, int(round(c * width)))
    if even and v % 2:
        v += 1
    return v


def student_config(width: float, in_chans: int = 11, num_classes: int = 15) -> Config:
    """Official config with channel widths scaled by ``width``."""
    cfg = Config.fromfile(CONFIG)
    cfg.model.backbone.in_chans = in_chans
    cfg.model.decode_head.num_classes = num_classes
    if width != 1.0:
        embed = [_scale(c, width, even=(i == 0)) for i, c in enumerate(TEACHER_EMBED)]
        dec = _scale(TEACHER_DECODER, width)
        cfg.model.backbone.embed_dims = embed
        cfg.model.decode_head.in_channels = embed
        cfg.model.decode_head.channels = dec
        cfg.model.decode_head.ham_channels = dec
        cfg.model.decode_head.norm_cfg.num_groups = math.gcd(GN_GROUPS, dec)
    return cfg


class MariNextStudent(nn.Module):
    """Same structure and attribute names as the official ``MariNext`` wrapper (``backbone``, ``decode_head``)."""

    def __init__(self, width: float = 1.0, in_chans: int = 11, num_classes: int = 15):
        super().__init__()
        self.width = width
        self.cfg = student_config(width, in_chans, num_classes)
        model = build_segmentor(self.cfg.model)
        model.init_weights()
        model = revert_sync_batchnorm(model)
        self.backbone = model.backbone
        self.decode_head = model.decode_head

    def forward(self, x):
        return self.decode_head(self.backbone(x))


def build_student(width: float, in_chans: int = 11, num_classes: int = 15) -> nn.Module:
    """Width-scaled MariNeXt; ``width=1.0`` reproduces the teacher architecture exactly."""
    return MariNextStudent(width, in_chans, num_classes)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
