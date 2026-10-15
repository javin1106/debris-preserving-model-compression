"""Memory-mapped cache of the official MADOS arrays (identical values, faster start, shared page cache).

``build_cache`` runs the official ``MADOS`` loader once per split and saves ``X`` (float32, N×11×240×240) and
``y`` (int64, N×240×240, ignore = -1) as ``.npy``; it verifies the saved arrays equal the loader's arrays.
``MADOSCached`` is the official class with ``__init__`` replaced by ``np.load(..., mmap_mode='r')``; the official
``__getitem__`` / ``join_transform`` (NaN imputation, flips/rotations, standardisation) are inherited unchanged.
Several training processes can then share one copy of the data through the OS page cache.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .official import MADOS, bands_mean, split_dir


def cache_paths(cache_dir: str, split: str):
    return Path(cache_dir) / f"{split}_X.npy", Path(cache_dir) / f"{split}_y.npy", Path(cache_dir) / f"{split}_ids.txt"


def build_cache(data_path: str, cache_dir: str, split: str) -> None:
    """Load ``split`` with the official loader and write X/y/ids to ``cache_dir`` (verified)."""
    px, py, pid = cache_paths(cache_dir, split)
    if px.exists() and py.exists() and pid.exists():
        print(f"cache exists: {split}")
        return
    os.makedirs(cache_dir, exist_ok=True)
    ds = MADOS(data_path, split_dir(data_path), split)
    np.save(px.with_suffix(".tmp.npy"), ds.X)
    np.save(py.with_suffix(".tmp.npy"), ds.y)
    X = np.load(px.with_suffix(".tmp.npy"), mmap_mode="r")
    y = np.load(py.with_suffix(".tmp.npy"), mmap_mode="r")
    assert X.dtype == ds.X.dtype and X.shape == ds.X.shape and np.array_equal(X, ds.X, equal_nan=True)
    assert y.dtype == ds.y.dtype and np.array_equal(y, ds.y)
    os.replace(px.with_suffix(".tmp.npy"), px)
    os.replace(py.with_suffix(".tmp.npy"), py)
    pid.write_text("\n".join(map(str, np.atleast_1d(ds.ROIs_split))) + "\n")
    print(f"cached {split}: X {ds.X.shape} {ds.X.dtype}, y {ds.y.shape} {ds.y.dtype}")


class MADOSCached(MADOS):
    """Official MADOS dataset backed by the verified ``.npy`` cache (same items, same augmentation code)."""

    def __init__(self, cache_dir: str, mode: str = "train"):  # noqa: D107 — deliberately skips MADOS.__init__
        px, py, pid = cache_paths(cache_dir, mode)
        self.X = np.load(px, mmap_mode="r")
        self.y = np.load(py, mmap_mode="r")
        self.ROIs_split = np.array(pid.read_text().split())
        self.impute_nan = np.tile(bands_mean, (self.X.shape[-1], self.X.shape[-2], 1))
        self.mode = mode
        self.length = len(self.y)
        self.path = str(cache_dir)
        self.input_size = 240

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, np.asarray(target)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the verified .npy cache of MADOS splits.")
    ap.add_argument("--path", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    a = ap.parse_args()
    for s in a.splits:
        build_cache(a.path, a.cache, s)


if __name__ == "__main__":
    main()
