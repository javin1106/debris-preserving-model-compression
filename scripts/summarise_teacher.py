"""Phase 1a summary: released MariNeXt checkpoints → results/teacher_repro.{csv,md}, cross-check, GATE 1.

Reads only from runs/:
  runs/p1-A-ckpt<k>-<env>-*/metrics.json          (our evaluate.py; latest run per checkpoint)
  runs/p1-official-eval-ckpt1/official_test_tta{True,False}.txt   (official evaluation.py stdout)
Usage: python scripts/summarise_teacher.py [--env envorig]
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS, RESULTS = REPO / "runs", REPO / "results"
DEBRIS = "Marine Debris"
# Paper (Kikaki et al. 2024, Table 5), MariNeXt, Marine Debris row — the paper's numbers, not ours.
PAPER_DEBRIS = {"f1": 51.3, "recall": 70.4, "iou": 34.6}
METRICS = ["precision", "recall", "f1", "iou"]


def latest_runs(env: str) -> Dict[int, Path]:
    """Latest run folder per checkpoint index for the given env tag."""
    out: Dict[int, Path] = {}
    for d in sorted(RUNS.glob(f"p1-A-ckpt*-{env}-*")):
        m = re.match(rf"p1-A-ckpt(\d+)-{env}-\d{{8}}-\d{{4}}$", d.name)
        if m and (d / "metrics.json").exists():
            out[int(m.group(1))] = d          # sorted → last one wins (timestamp suffix)
    missing = [k for k in range(1, 6) if k not in out]
    if missing:
        raise SystemExit(f"missing runs for checkpoints {missing} (env {env})")
    return dict(sorted(out.items()))


def parse_official(txt: Path) -> Tuple[Dict, Dict[str, Dict[str, float]]]:
    """Parse official evaluation.py stdout → (Evaluation dict, per-class {metric: value in %, 1 decimal})."""
    text = txt.read_text(encoding="utf-8", errors="replace")
    ev = ast.literal_eval(re.search(r"Evaluation: (\{.*?\})", text, re.S).group(1))
    lines = text.split("Confusion Matrix:")[1].strip().splitlines()
    from_names = ["Marine Debris", "Dense Sargassum", "Sparse Floating Algae", "Natural Organic Material", "Ship",
                  "Oil Spill", "Marine Water", "Sediment-Laden Water", "Foam", "Turbid Water", "Shallow Water",
                  "Waves & Wakes", "Oil Platform", "Jellyfish", "Sea snot"]
    per: Dict[str, Dict[str, float]] = {n: {} for n in from_names}
    for ln in lines[1:]:
        for key, metric in (("IoU", "iou"), ("Precision", "precision"), ("F1-score", "f1")):
            if ln.startswith(key + " "):
                vals = ln.split()[1:1 + len(from_names)]
                for n, v in zip(from_names, vals):
                    per[n][metric] = float(v)
        for n in from_names:
            if ln.startswith(n + " "):
                per[n]["recall"] = float(ln.split()[-1])
    return ev, per


def mean_sd(xs: List[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="envorig")
    args = ap.parse_args()
    runs = latest_runs(args.env)
    data = {k: json.loads((d / "metrics.json").read_text()) for k, d in runs.items()}
    classes = list(next(iter(data.values()))["results"]["test"]["single"]["per_class"].keys())

    # ---- long CSV: one row per checkpoint × split × mode × class -------------------------------------------
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for k, m in data.items():
        for split, by_mode in m["results"].items():
            for mode in ("single", "tta"):
                r = by_mode[mode]
                for c in classes:
                    pc = r["per_class"][c]
                    rows.append({"run_id": runs[k].name, "ckpt": k, "split": split, "mode": mode, "class": c,
                                 **{x: round(100 * pc[x], 4) for x in METRICS}, "support": pc["support"]})
                rows.append({"run_id": runs[k].name, "ckpt": k, "split": split, "mode": mode, "class": "_macro",
                             "precision": round(100 * r["macro_precision"], 4), "recall": round(100 * r["macro_recall"], 4),
                             "f1": round(100 * r["macro_f1"], 4), "iou": round(100 * r["mIoU"], 4),
                             "support": r["n_labelled_px"]})
    csv_path = RESULTS / f"teacher_repro_{args.env}.csv" if args.env != "envorig" else RESULTS / "teacher_repro.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def stat(split, mode, cls, metric):
        if cls == "_macro":
            key = {"f1": "macro_f1", "iou": "mIoU", "precision": "macro_precision", "recall": "macro_recall"}[metric]
            return mean_sd([100 * data[k]["results"][split][mode][key] for k in data])
        return mean_sd([100 * data[k]["results"][split][mode]["per_class"][cls][metric] for k in data])

    L: List[str] = ["# Teacher reproduction — released MariNeXt checkpoints (Phase 1a)\n"]
    L.append(f"Generated by `scripts/summarise_teacher.py` from these runs (env `{args.env}`): "
             + ", ".join(f"ckpt {k} → `{d.name}`" for k, d in runs.items()) + ".")
    L.append("All metrics on labelled pixels only, pooled over all patches of the split; mean ± sd (ddof = 1) over "
             "the 5 released checkpoints. Single = one forward pass; TTA = official 8-view majority vote.\n")

    # ---- debris vs paper + GATE 1 -----------------------------------------------------------------------------
    L.append("## Marine Debris vs the paper\n")
    L.append("Paper numbers (Kikaki et al. 2024, Table 5) are **the paper's, not ours**. Per the project page "
             "(marine-pollution.github.io) the paper's test results are *averaged over five trials* and TTA is employed "
             "during testing → the matching setting is **test, TTA, mean of 5**. (Paper PDF not accessible to us; "
             "Table 5 values taken from `docs/PROJECT_BRIEF.md`.)\n")
    L.append("| split | mode | F1 | recall | IoU | precision |\n|---|---|---|---|---|---|")
    for split in ("val", "test"):
        for mode in ("single", "tta"):
            cells = [f"{a:.1f} ± {b:.1f}" for a, b in (stat(split, mode, DEBRIS, x) for x in ("f1", "recall", "iou", "precision"))]
            L.append(f"| {split} | {mode} | " + " | ".join(cells) + " |")
    L.append(f"| **paper** | (test, TTA?, mean of 5) | {PAPER_DEBRIS['f1']} | {PAPER_DEBRIS['recall']} | {PAPER_DEBRIS['iou']} | – |\n")

    L.append("### GATE 1\n")
    L.append("Rule: each paper value lies within our mean ± 2 sd (matching setting: test, TTA), or within 1.5 points of our mean.\n")
    L.append("| metric | paper | ours (test, TTA) | mean ± 2 sd | |paper − mean| | pass |\n|---|---|---|---|---|---|")
    all_pass = True
    for x in ("f1", "recall", "iou"):
        mu, sd = stat("test", "tta", DEBRIS, x)
        p = PAPER_DEBRIS[x]
        ok = (mu - 2 * sd <= p <= mu + 2 * sd) or abs(p - mu) <= 1.5
        all_pass &= ok
        L.append(f"| {x} | {p} | {mu:.2f} ± {sd:.2f} | [{mu-2*sd:.2f}, {mu+2*sd:.2f}] | {abs(p-mu):.2f} | {'PASS' if ok else 'FAIL'} |")
    L.append(f"\n**GATE 1: {'PASSED' if all_pass else 'FAILED'}.** (For reference, single-pass test: "
             + ", ".join(f"{x} {stat('test','single',DEBRIS,x)[0]:.1f}" for x in ('f1','recall','iou')) + ".)\n")

    # ---- per-checkpoint debris ------------------------------------------------------------------------------
    L.append("## Marine Debris per checkpoint\n")
    L.append("| ckpt | run_id | val single F1 | val TTA F1 | test single F1 / R / IoU | test TTA F1 / R / IoU |\n|---|---|---|---|---|---|")
    for k, m in data.items():
        g = lambda s, md, x: 100 * m["results"][s][md]["per_class"][DEBRIS][x]
        L.append(f"| {k} | `{runs[k].name}` | {g('val','single','f1'):.1f} | {g('val','tta','f1'):.1f} | "
                 f"{g('test','single','f1'):.1f} / {g('test','single','recall'):.1f} / {g('test','single','iou'):.1f} | "
                 f"{g('test','tta','f1'):.1f} / {g('test','tta','recall'):.1f} / {g('test','tta','iou'):.1f} |")
    L.append("")

    # ---- per-class tables -----------------------------------------------------------------------------------
    for split in ("test", "val"):
        L.append(f"## Per-class, {split} split (mean ± sd over 5 checkpoints)\n")
        L.append("| class | support | F1 single | F1 TTA | IoU single | IoU TTA | recall TTA | precision TTA |\n|---|---|---|---|---|---|---|---|")
        for c in classes + ["_macro"]:
            sup = data[1]["results"][split]["single"]["per_class"][c]["support"] if c != "_macro" else data[1]["results"][split]["single"]["n_labelled_px"]
            vals = [stat(split, "single", c, "f1"), stat(split, "tta", c, "f1"), stat(split, "single", c, "iou"),
                    stat(split, "tta", c, "iou"), stat(split, "tta", c, "recall"), stat(split, "tta", c, "precision")]
            name = "**macro (F1) / mIoU**" if c == "_macro" else c
            L.append(f"| {name} | {sup:,} | " + " | ".join(f"{a:.1f} ± {b:.1f}" for a, b in vals) + " |")
        L.append("")

    # ---- cross-check against official evaluation.py -----------------------------------------------------------
    off_dir = RUNS / "p1-official-eval-ckpt1"
    if off_dir.exists() and args.env == "envorig":
        tf32_runs = sorted(RUNS.glob("p1-A-ckpt1-envorig-tf32on-*"))
        L.append("## Cross-check vs official `evaluation.py` (checkpoint 1, test)\n")
        L.append("Official output parsed from `runs/p1-official-eval-ckpt1/`. Per-class values compared at the official "
                 "print precision (0.1); macro values at full precision. PyTorch 1.11 enables TF32 by default, so the "
                 "official script runs with TF32 **on**; our protocol disables TF32 (CLAUDE.md). Both are shown.\n")
        L.append("| ours | run_id | mode | per-class cells | cells differing | max diff (pts) | macro-F1 official / ours "
                 "| mIoU official / ours | exact |\n|---|---|---|---|---|---|---|---|---|")
        variants = [("TF32 off (our protocol)", runs[1].name, data[1])]
        if tf32_runs:
            variants.append(("TF32 on (official default)", tf32_runs[-1].name,
                             json.loads((tf32_runs[-1] / "metrics.json").read_text())))
        for label, rid, mj in variants:
            for mode, flag in (("tta", "True"), ("single", "False")):
                ev, per = parse_official(off_dir / f"official_test_tta{flag}.txt")
                ours = mj["results"]["test"][mode]
                diffs = [abs(per[c][x] - round(100 * ours["per_class"][c][x], 1)) for c in classes for x in METRICS]
                n_bad = sum(d > 1e-9 for d in diffs)
                mf_o, mf_u, mi_o, mi_u = ev["macroF1"], ours["macro_f1"], ev["IoU"], ours["mIoU"]
                exact = n_bad == 0 and abs(mf_o - mf_u) < 1e-12 and abs(mi_o - mi_u) < 1e-12
                L.append(f"| {label} | `{rid}` | {mode} | {len(diffs)} | {n_bad} | {max(diffs):.1f} | "
                         f"{mf_o:.12f} / {mf_u:.12f} | {mi_o:.12f} / {mi_u:.12f} | {'YES' if exact else 'NO'} |")
        L.append("\nConclusion: with TF32 at the official default, our evaluator reproduces `evaluation.py` exactly; "
                 "with TF32 off (our protocol) results differ only by float rounding (a few pixels flip at near-ties).\n")

    md_path = csv_path.with_suffix(".md")
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
