"""Single place that puts the official MADOS code on ``sys.path`` and re-exports what we use.

Nothing here changes official behaviour; ``seed_all`` is a verbatim copy of the function that
``marinext/train.py`` and ``marinext/evaluation.py`` both define.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MADOS_DIR = REPO_ROOT / "third_party" / "mados"
for _p in (MADOS_DIR / "marinext", MADOS_DIR / "utils"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from assets import labels as CLASS_NAMES  # noqa: E402
from dataset import MADOS, bands_mean, bands_std, class_distr, gen_weights  # noqa: E402
from marinext_wrapper import MariNext  # noqa: E402
from test_time_aug import TTA  # noqa: E402

NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = -1                              # loader does y - 1 on uint8 labels where 0 = unlabelled
DEBRIS = CLASS_NAMES.index("Marine Debris")    # = 0


def seed_all(seed: int) -> None:
    """Verbatim copy of the official ``seed_all`` (torch, cuda, numpy, random; deterministic cuDNN)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_official_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> None:
    """Load a state dict exactly as ``evaluation.py`` does: rename ``decoder``→``decode_head``, drop proj1/proj2."""
    sd: Dict[str, torch.Tensor] = torch.load(path, map_location=device)
    sd = {k.replace("decoder", "decode_head"): v for k, v in sd.items() if ("proj1" not in k) and ("proj2" not in k)}
    model.load_state_dict(sd)


def split_dir(data_path: str) -> str:
    return os.path.join(data_path, "splits")
