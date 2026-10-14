"""Count MariNeXt parameters, built through the official wrapper and config (Phase 0.5)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
MARINEXT_DIR = REPO_ROOT / "third_party" / "mados" / "marinext"
ANALYTIC_TEACHER_PARAMS = 4_238_063


def import_marinext():
    """Import the official ``MariNext`` wrapper class from the submodule (vendored mmseg sits next to it)."""
    if str(MARINEXT_DIR) not in sys.path:
        sys.path.insert(0, str(MARINEXT_DIR))
    from marinext_wrapper import MariNext  # noqa: E402

    return MariNext


def count_params(model) -> Dict[str, int]:
    """Return total and trainable parameter counts of ``model``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_chans", type=int, default=11)
    parser.add_argument("--num_classes", type=int, default=15)
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()

    MariNext = import_marinext()
    model = MariNext(args.in_chans, args.num_classes)
    counts = count_params(model)
    counts["analytic"] = ANALYTIC_TEACHER_PARAMS
    counts["diff_vs_analytic"] = counts["total"] - ANALYTIC_TEACHER_PARAMS
    counts["config"] = os.path.relpath(MARINEXT_DIR / "configs" / "marinext.tiny.240x240.mados.py", REPO_ROOT)

    if args.json:
        print(json.dumps(counts))
        return
    print(f"MariNeXt ({args.in_chans} bands, {args.num_classes} classes)")
    print(f"  total params     : {counts['total']:,}")
    print(f"  trainable params : {counts['trainable']:,}")
    print(f"  analytic value   : {ANALYTIC_TEACHER_PARAMS:,}  (diff {counts['diff_vs_analytic']:+,})")


if __name__ == "__main__":
    main()
