"""Run IDs, run folders and env.json, following the conventions in CLAUDE.md."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def disable_tf32() -> None:
    """Disable TF32 for parity across GPUs (CLAUDE.md)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_run_id(phase: str, setting: str, tag: str) -> str:
    """``<phase>-<setting>-<tag>-<YYYYMMDD-HHMM>``; tag is ``w<width>-s<seed>`` for trained models."""
    return f"{phase}-{setting}-{tag}-{datetime.now().strftime('%Y%m%d-%H%M')}"


def run_dir(run_id: str) -> Path:
    d = REPO_ROOT / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _git(args, cwd: Path) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def env_info() -> Dict:
    """Python/torch/CUDA/mmcv versions, GPU, TF32 flags, git hashes of this repo and the submodule."""
    try:
        import mmcv
        mmcv_v = mmcv.__version__
    except Exception:
        mmcv_v = None
    sub = REPO_ROOT / "third_party" / "mados"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "mmcv": mmcv_v,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "repo_commit": _git(["rev-parse", "HEAD"], REPO_ROOT),
        "repo_dirty": bool(_git(["status", "--porcelain", "--", "src", "scripts", "configs"], REPO_ROOT)),
        "submodule_commit": _git(["rev-parse", "HEAD"], sub),
        "submodule_dirty": bool(_git(["status", "--porcelain"], sub)),
        "glibc_tunables": os.environ.get("GLIBC_TUNABLES"),
    }


def write_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
