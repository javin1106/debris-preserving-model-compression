"""Phase 0.4 data report: read MADOS through the official loader and write results/data_report.md.

Every number in the report is computed here from the data on disk or from the official code's constants.
Usage (env-orig, repo root):  python scripts/inspect_data.py --path data/MADOS
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MADOS_DIR = REPO_ROOT / "third_party" / "mados"
sys.path.insert(0, str(MADOS_DIR / "utils"))
sys.path.insert(0, str(MADOS_DIR / "marinext"))

import dataset as off_dataset  # noqa: E402  official utils/dataset.py
from assets import labels as OFF_LABELS, mados_cat_mapping  # noqa: E402
from vscp import VSCP  # noqa: E402

SPLITS = ["train", "val", "test"]


def raw_label_values(path: str, max_files: int = 0) -> Dict[int, int]:
    """Histogram of raw on-disk values in the *_cl_* label rasters (before the loader's ``y - 1``)."""
    from osgeo import gdal

    files = sorted(glob(os.path.join(path, "*", "10", "*_L2R_cl_*.tif")))
    if max_files:
        files = files[:max_files]
    hist: Dict[int, int] = {}
    for f in files:
        ds = gdal.Open(f)
        arr = ds.ReadAsArray()
        ds = None
        vals, cnts = np.unique(arr, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            hist[int(v)] = hist.get(int(v), 0) + int(c)
    return dict(sorted(hist.items())), len(files), str(arr.dtype)


def band_files_of_first_crop(path: str) -> List[str]:
    """Band file names of one crop, in the order the official loader stacks them."""
    tile = [t for t in sorted(glob(os.path.join(path, "*"))) if os.path.isdir(os.path.join(t, "10"))][0]
    crop = sorted(f.split("_cl_")[-1] for f in glob(os.path.join(tile, "10", "*_cl_*")))[0]
    bands = sorted(glob(os.path.join(tile, "*", "*L2R_rhorc*_" + crop)), key=off_dataset.get_band)
    return [os.path.relpath(b, path).replace(os.sep, "/") for b in bands]


def split_stats(path: str, split: str) -> Dict:
    """Load one split through the official ``MADOS`` class and compute its statistics."""
    ds = off_dataset.MADOS(path, os.path.join(path, "splits"), split)
    X, y = ds.X, ds.y
    n_classes = len(OFF_LABELS)
    lab = y[y != -1]
    counts = np.bincount(lab.ravel(), minlength=n_classes)[:n_classes]
    stats = {
        "n_patches": int(len(ds)),
        "n_split_ids": int(np.size(ds.getnames())),
        "X_shape": list(X.shape),
        "X_dtype": str(X.dtype),
        "y_dtype": str(y.dtype),
        "y_unique": sorted(int(v) for v in np.unique(y)),
        "nan_fraction_X": float(np.isnan(X).mean()),
        "pixels_total": int(y.size),
        "pixels_labelled": int(lab.size),
        "unlabelled_fraction": float((y == -1).mean()),
        "class_counts": [int(c) for c in counts],
        "patches_with_debris": int(((y == 0).reshape(len(y), -1).any(1)).sum()),
    }
    if split == "train":
        img, tgt = ds[0]
        stats["getitem_image"] = {"shape": list(img.shape), "dtype": str(img.dtype)}
        stats["getitem_target"] = {"shape": list(tgt.shape), "dtype": str(tgt.dtype)}
        stats["vscp_demo"] = vscp_demo(ds)
    del ds, X, y, lab
    gc.collect()
    return stats


def vscp_demo(ds) -> Dict:
    """Run the official VSCP on a fixed batch of 5 training items and measure what it changes."""
    rng = np.random.RandomState(0)
    idx = rng.choice(len(ds), 5, replace=False)
    items = [ds[i] for i in idx]
    img = np.stack([a for a, _ in items])
    tgt = np.stack([b for _, b in items])
    aug_img, aug_tgt = VSCP(img.copy(), tgt.copy())
    n_aug = img.shape[0] // 2
    out = {"batch_in": int(img.shape[0]), "batch_added": int(aug_img.shape[0]), "per_pair": []}
    for i in range(n_aug):
        src_mask = tgt[i + n_aug] != -1
        out["per_pair"].append({
            "target_img": i, "source_img": i + n_aug,
            "source_labelled_px": int(src_mask.sum()),
            "labels_changed_px": int((aug_tgt[i] != tgt[i]).sum()),
            "pasted_labels_match_source": bool((aug_tgt[i][src_mask] == tgt[i + n_aug][src_mask]).all()),
            "pasted_bands_match_source": bool(np.allclose(aug_img[i][:, src_mask], img[i + n_aug][:, src_mask])),
            "unpasted_px_unchanged": bool((aug_tgt[i][~src_mask] == tgt[i][~src_mask]).all()
                                          and np.allclose(aug_img[i][:, ~src_mask], img[i][:, ~src_mask])),
            "labelled_px_before": int((tgt[i] != -1).sum()),
            "labelled_px_after": int((aug_tgt[i] != -1).sum()),
        })
    return out


def model_size() -> Dict:
    """Run count_params.py in a subprocess (keeps the mmseg import out of this process) and parse JSON."""
    res = subprocess.run([sys.executable, str(REPO_ROOT / "src" / "debriskd" / "count_params.py"), "--json"],
                         capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip().splitlines()[-1])


def fmt_int(x: int) -> str:
    return f"{x:,}"


def write_report(path: str, out: Path, raw, bands, per_split, msize) -> None:
    hist, n_label_files, raw_dtype = raw
    L: List[str] = []
    L.append("# MADOS data report (Phase 0.4)\n")
    L.append(f"Generated by `scripts/inspect_data.py` from `{path}` using the official loader "
             "`third_party/mados/utils/dataset.py` (submodule commit in `docs/ENVIRONMENT.md`).\n")

    L.append("## Classes and indices\n")
    L.append("Source: `utils/assets.py` (`mados_cat_mapping`, on-disk value) and `utils/dataset.py` "
             "(`self.y = self.y - 1`, model index).\n")
    L.append("| model index | on-disk value | class |\n|---|---|---|")
    for i, name in enumerate(OFF_LABELS):
        L.append(f"| {i} | {mados_cat_mapping[name]} | {name} |")
    debris_idx = OFF_LABELS.index("Marine Debris")
    L.append(f"\n**Marine Debris = model index {debris_idx}** (on-disk value {mados_cat_mapping['Marine Debris']}).\n")

    L.append("## Ignore index\n")
    L.append(f"- Raw label rasters (`*_L2R_cl_*.tif`, {n_label_files} files, dtype `{raw_dtype}`) contain values: "
             + ", ".join(f"{k}: {fmt_int(v)} px" for k, v in hist.items()) + ".")
    L.append("- The loader subtracts 1, so on-disk **0 (unlabelled) becomes -1**. Training uses "
             "`CrossEntropyLoss(ignore_index=-1)` and evaluation masks `target != -1`.")
    uniq = sorted(set().union(*[set(s["y_unique"]) for s in per_split.values()]))
    L.append(f"- Label values after loading, all splits: {uniq}.")
    L.append(f"\n**Ignore index = -1** (as loaded); unlabelled pixels are stored as 0 on disk.\n")

    L.append("## Bands, patch size, dtype, normalisation\n")
    L.append(f"- {len(bands)} bands per patch, stacked in this order (sorted by the wavelength number in the file name; "
             "folder = native resolution in m, upsampled to 10 m by nearest neighbour):")
    for i, b in enumerate(bands):
        L.append(f"  {i}. `{b}`")
    tr = per_split["train"]
    L.append(f"- Stored array (train): shape {tr['X_shape']} (N, bands, H, W), dtype `{tr['X_dtype']}`; "
             f"labels dtype `{tr['y_dtype']}`.")
    L.append(f"- `__getitem__` returns image {tr['getitem_image']['shape']} `{tr['getitem_image']['dtype']}`, "
             f"target {tr['getitem_target']['shape']} `{tr['getitem_target']['dtype']}`.")
    L.append("- NaNs are replaced by the per-band mean (`bands_mean`); then per-band standardisation "
             "`(x - bands_mean) / bands_std` with the hard-coded constants below.")
    L.append("- NaN fraction of raw pixel values: " + ", ".join(
        f"{s}: {per_split[s]['nan_fraction_X']:.4%}" for s in SPLITS) + ".")
    L.append("\n| band | mean | std |\n|---|---|---|")
    for i, (m, s) in enumerate(zip(off_dataset.bands_mean, off_dataset.bands_std)):
        L.append(f"| {i} | {m:.8f} | {s:.8f} |")
    L.append("\nTrain-only augmentation in the loader: random flip (h / v / both / none×2), then with p = 0.8 a "
             "rotation by k·90°, k ∈ {0,1,2,3}. Val/test are not augmented.\n")

    L.append("## Patches per split\n")
    L.append("| split | patches loaded | ids in split file | patches containing debris |\n|---|---|---|---|")
    for s in SPLITS:
        p = per_split[s]
        L.append(f"| {s} | {p['n_patches']} | {p['n_split_ids']} | {p['patches_with_debris']} |")
    L.append("")

    L.append("## Labelled pixels per class per split\n")
    L.append("| idx | class | " + " | ".join(SPLITS) + " | share of labelled (train) |\n|---|---|"
             + "---|" * len(SPLITS) + "---|")
    for i, name in enumerate(OFF_LABELS):
        row = [fmt_int(per_split[s]["class_counts"][i]) for s in SPLITS]
        share = per_split["train"]["class_counts"][i] / max(1, per_split["train"]["pixels_labelled"])
        L.append(f"| {i} | {name} | " + " | ".join(row) + f" | {share:.4%} |")
    L.append("| | **labelled total** | " + " | ".join(fmt_int(per_split[s]["pixels_labelled"]) for s in SPLITS) + " | |")
    L.append("| | **all pixels** | " + " | ".join(fmt_int(per_split[s]["pixels_total"]) for s in SPLITS) + " | |")
    L.append("| | **unlabelled fraction** | " + " | ".join(f"{per_split[s]['unlabelled_fraction']:.4%}" for s in SPLITS) + " | |")
    L.append("\nFor comparison, the hard-coded `class_distr` used for the loss weights in `utils/dataset.py`: "
             + ", ".join(f"{v:.5f}" for v in off_dataset.class_distr.tolist()) + ".\n")

    v = per_split["train"]["vscp_demo"]
    L.append("## VSCP augmentation (`utils/vscp.py`)\n")
    L.append("Called in `train.py` on every training batch after loader augmentation. With batch size B it builds "
             "`B // 2` new images: for i < B//2, every **labelled** pixel of image i + B//2 (all 11 bands and the label) "
             "is copied onto image i. Unlabelled pixels of the source are not copied, so the target image keeps its own "
             "bands and labels there. The new images are appended to the batch (B = 5 → 7 images per step); the originals "
             "are kept unchanged.\n")
    L.append(f"Check on a fixed batch of 5 training items (seed 0): {v['batch_in']} in → {v['batch_added']} added.\n")
    L.append("| target ← source | source labelled px | labels changed px | pasted labels = source | pasted bands = source "
             "| other px unchanged | labelled px before → after |\n|---|---|---|---|---|---|---|")
    for p in v["per_pair"]:
        L.append(f"| {p['target_img']} ← {p['source_img']} | {fmt_int(p['source_labelled_px'])} | "
                 f"{fmt_int(p['labels_changed_px'])} | {p['pasted_labels_match_source']} | {p['pasted_bands_match_source']} | "
                 f"{p['unpasted_px_unchanged']} | {fmt_int(p['labelled_px_before'])} → {fmt_int(p['labelled_px_after'])} |")
    L.append("")

    L.append("## Model size\n")
    L.append(f"Built with the official wrapper `marinext/marinext_wrapper.py` and config `{msize['config']}` "
             "(11 bands, 15 classes), counted by `src/debriskd/count_params.py`.\n")
    L.append(f"- Total parameters: **{fmt_int(msize['total'])}**; trainable: {fmt_int(msize['trainable'])}.")
    L.append(f"- Analytic value from `CLAUDE.md`: {fmt_int(msize['analytic'])}; difference {msize['diff_vs_analytic']:+,}.")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(REPO_ROOT / "data" / "MADOS"))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "data_report.md"))
    parser.add_argument("--json_out", default=str(REPO_ROOT / "results" / "data_report.json"))
    args = parser.parse_args()

    raw = raw_label_values(args.path)
    bands = band_files_of_first_crop(args.path)
    per_split = {s: split_stats(args.path, s) for s in SPLITS}
    msize = model_size()
    write_report(args.path, Path(args.out), raw, bands, per_split, msize)
    Path(args.json_out).write_text(json.dumps({"raw_label_hist": raw[0], "bands": bands, "splits": per_split,
                                               "model_size": msize}, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
