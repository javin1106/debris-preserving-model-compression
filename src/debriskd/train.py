# -*- coding: utf-8 -*-
"""Training script for MariNeXt teachers and width-scaled students (Phases 1c, 2; KD added in Phase 3).

Copied from the official ``third_party/mados/marinext/train.py`` at submodule commit
c20a7e972bdf111126540d8e6caa45808352f138 (Ioannis Kakogeorgiou, MIT). The training recipe is unchanged:
Adam lr 2e-4, no weight decay, MultiStepLR [45, 65] ×0.1 stepped once per epoch, 80 epochs, batch 5 + VSCP
(→ 7 images/step), class-weighted CE (``1/ln(1.03 + class_distr)``, ignore -1), timm ``ModelEma`` decay 0.999
updated every step, loader flips/rotations, validation every epoch on raw and EMA weights.

Additions (no effect on the optimisation itself):
* ``--width``: build a width-scaled student (``students.build_student``); 1.0 = teacher architecture.
* ``--seed``: replaces the official hard-coded ``seed_all(0)`` / ``Generator().manual_seed(0)``, including the
  re-seed before every validation pass. ``--seed 0`` therefore reproduces the official RNG behaviour.
* Run folder ``runs/<run_id>/`` (config.yaml, train.log, epochs.jsonl, metrics.json, env.json), TF32 disabled.
* Full resume (model, EMA, optimiser, scheduler, RNG states) from ``resume.pth``, written every epoch.
* Model selection (decision 2026-10-15): EMA weights of the epoch with the best validation macro-F1
  (ties → earlier epoch), saved as ``best_ema.pth``; then evaluated once on val and test (single + TTA)
  with the same protocol as ``evaluate.py``.
* Optional ``--cache`` (verified memory-mapped copy of the official arrays) and ``--subset`` for smoke tests.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import random
import sys
import time
from typing import Dict

import numpy as np
import torch
import yaml
from timm.utils import ModelEma
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .kd_loss import KDConfig, lam_U_at, total_loss
from .metrics import confusion, scores_from_confusion
from .official import (CLASS_NAMES, DEBRIS, IGNORE_INDEX, MADOS, NUM_CLASSES, MariNext, class_distr, gen_weights,
                       load_official_checkpoint, seed_all, split_dir)
from .runlog import disable_tf32, env_info, make_run_id, run_dir, write_json
from .students import build_student, count_params

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "mados", "utils"))
from assets import bool_flag, cosine_scheduler  # noqa: E402
from vscp import VSCP  # noqa: E402


def seed_worker(worker_id):
    # DataLoader Workers Reproducibility (official)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataset(options, mode):
    if options["cache"]:
        from .data import MADOSCached
        ds = MADOSCached(options["cache"], mode)
    else:
        ds = MADOS(options["path"], split_dir(options["path"]), mode)
    if options["subset"]:
        ds = Subset(ds, range(min(options["subset"], len(ds))))
    return ds


@torch.no_grad()
def validate(model, loader, criterion, device, seed) -> Dict:
    """Official validation loop (batch = --batch, one re-seed before the loop), metrics on labelled pixels."""
    model.eval()
    seed_all(seed)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    loss_sum, n_px = 0.0, 0
    for image, target in loader:
        image, target = image.to(device), target.to(device)
        logits = F.interpolate(model(image), size=(target.shape[-2], target.shape[-1]), mode="bilinear",
                               align_corners=False)
        loss = criterion(logits, target)
        pred = logits.argmax(1)
        n = int((target != IGNORE_INDEX).sum())
        loss_sum += float(loss) * n
        n_px += n
        cm += confusion(pred.cpu().numpy(), target.cpu().numpy(), NUM_CLASSES, IGNORE_INDEX)
    s = scores_from_confusion(cm, CLASS_NAMES)
    s["loss"] = loss_sum / max(n_px, 1)
    return s


def short(s: Dict) -> Dict:
    d = s["per_class"][CLASS_NAMES[DEBRIS]]
    return {"loss": s["loss"], "macro_f1": s["macro_f1"], "mIoU": s["mIoU"], "debris_f1": d["f1"],
            "debris_recall": d["recall"], "debris_iou": d["iou"]}


def main(options):
    disable_tf32()
    rd = run_dir(options["run_id"])
    ckpt_dir = rd / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    logging.basicConfig(filename=str(rd / "train.log"), filemode="a", level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    (rd / "config.yaml").write_text(yaml.safe_dump(options, sort_keys=False), encoding="utf-8")
    write_json(rd / "env.json", env_info())

    seed = options["seed"]
    seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    dataset_train = make_dataset(options, "train")
    dataset_val = make_dataset(options, "val")
    loader_kw = dict(batch_size=options["batch"], num_workers=options["num_workers"], pin_memory=options["pin_memory"],
                     worker_init_fn=seed_worker, generator=g)
    if options["num_workers"] > 0:
        loader_kw.update(prefetch_factor=options["prefetch_factor"], persistent_workers=options["persistent_workers"])
    train_loader = DataLoader(dataset_train, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(dataset_val, shuffle=False, **loader_kw)

    device = torch.device("cuda")
    model = build_student(options["width"], options["input_channels"], options["output_channels"]).to(device)
    logging.info("width %s, params %d", options["width"], count_params(model))

    model_ema = ModelEma(model, decay=options["model_ema_decay"], device=device, resume="")
    ema_decay_schedule = cosine_scheduler(options["model_ema_decay"], 0.999, options["epochs"], len(train_loader))

    weight = gen_weights(class_distr, c=options["weight_param"])
    if options["extra_debris_ce_weight"] != 1.0:          # setting C: extra debris-focused reweighting
        weight[DEBRIS] = weight[DEBRIS] * options["extra_debris_ce_weight"]
    logging.info("CE class weights: %s", [round(float(w), 4) for w in weight])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="mean", weight=weight.to(device),
                                          label_smoothing=options["label_smoothing"])
    optimizer = torch.optim.Adam(model.parameters(), lr=options["lr"], weight_decay=options["decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, options["lr_steps"], gamma=0.1)

    teacher, kd_cfg = None, None
    if options["teacher_ckpt"]:
        teacher = MariNext(options["input_channels"], options["output_channels"]).to(device)
        load_official_checkpoint(teacher, options["teacher_ckpt"], device)
        teacher.eval()
        teacher.requires_grad_(False)
        kd_cfg = KDConfig(T=options["kd_T"], tau=options["kd_tau"], lam_L=options["lam_L"], lam_U=options["lam_U"],
                          debris_w=options["debris_w"], debris_w_max=options["debris_w_max"],
                          warmup_epochs=options["kd_warmup_epochs"],
                          class_weights=weight.clone() if options["kd_class_weighted"] else None)
        logging.info("teacher %s, KD %s", options["teacher_ckpt"], kd_cfg)

    start, best = 1, {"epoch": None, "macro_f1": -1.0}
    resume_file = rd / "resume.pth"
    if resume_file.exists():
        st = torch.load(resume_file, map_location="cpu")  # RNG states must stay CPU ByteTensors
        model.load_state_dict(st["model"]); model_ema.ema.load_state_dict(st["ema"])
        optimizer.load_state_dict(st["optimizer"]); scheduler.load_state_dict(st["scheduler"])
        torch.set_rng_state(st["rng_torch"]); torch.cuda.set_rng_state(st["rng_cuda"])
        np.random.set_state(st["rng_numpy"]); random.setstate(st["rng_random"]); g.set_state(st["rng_loader"])
        start, best = st["epoch"] + 1, st["best"]
        logging.info("resumed from epoch %d (best so far %s)", st["epoch"], best)

    epochs = options["epochs"]
    for epoch in range(start, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, n_img = 0.0, 0
        kd_sum = {"sup": 0.0, "kd_lab": 0.0, "kd_ign": 0.0, "frac_ign_used": 0.0}
        kd_hist = torch.zeros(NUM_CLASSES, dtype=torch.long)
        for it, (image, target) in enumerate(train_loader):
            it = len(train_loader) * (epoch - 1) + it  # global training iteration
            if options["vscp"]:
                image_augmented, target_augmented = VSCP(image.cpu().numpy(), target.cpu().numpy())
                image = torch.cat([image, torch.tensor(image_augmented).to(image.device)])
                target = torch.cat([target, torch.tensor(target_augmented).to(target.device)])
            image = image.cuda()
            target = target.long().cuda()
            optimizer.zero_grad()
            logits = F.interpolate(model(image), size=image.size()[2:4], mode="bilinear", align_corners=False)
            loss = criterion(logits, target)
            if teacher is not None:
                # Teacher runs live on the same augmented batch. fork_rng: the teacher's random NMF bases do not
                # consume the training RNG stream, so the student sees exactly the same randomness as in setting B.
                with torch.no_grad(), torch.random.fork_rng(devices=[device]):
                    t_logits = F.interpolate(teacher(image), size=image.size()[2:4], mode="bilinear",
                                             align_corners=False)
                out = total_loss(loss, logits, t_logits, target, kd_cfg, epoch)
                loss = out["loss"]
                for k in kd_sum:
                    kd_sum[k] += float(out.get(k, 0.0))
                if "teacher_hist_ign" in out:
                    kd_hist += out["teacher_hist_ign"]
            loss.backward()
            if options["clip_grad"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), options["clip_grad"])
            optimizer.step()
            model_ema.decay = ema_decay_schedule[it]
            model_ema.update(model)
            loss_sum += float(loss) * target.shape[0]
            n_img += target.shape[0]
        t_train = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        val_raw = validate(model, val_loader, criterion, device, seed)
        scheduler.step()
        val_ema = validate(model_ema.ema, val_loader, criterion, device, seed)

        torch.save(model_ema.ema.state_dict(), ckpt_dir / "last_ema.pth")
        if val_ema["macro_f1"] > best["macro_f1"]:
            best = {"epoch": epoch, "macro_f1": val_ema["macro_f1"]}
            torch.save(model_ema.ema.state_dict(), ckpt_dir / "best_ema.pth")
        rec = {"epoch": epoch, "lr": lr, "train_loss": loss_sum / max(n_img, 1), "val_raw": short(val_raw),
               "val_ema": short(val_ema), "best_epoch": best["epoch"], "t_train_s": t_train,
               "t_epoch_s": time.time() - t0}
        if teacher is not None:
            nb = len(train_loader)
            rec["kd"] = {k: v / nb for k, v in kd_sum.items()}
            rec["kd"]["lam_U"] = lam_U_at(epoch, kd_cfg)
            rec["kd"]["teacher_hist_ign"] = {CLASS_NAMES[i]: int(c) for i, c in enumerate(kd_hist.tolist()) if c}
        with open(rd / "epochs.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        logging.info("epoch %d | train %.4f | val raw mF1 %.4f dF1 %.4f | val ema mF1 %.4f dF1 %.4f | best %s | %.0fs",
                     epoch, rec["train_loss"], val_raw["macro_f1"], rec["val_raw"]["debris_f1"],
                     val_ema["macro_f1"], rec["val_ema"]["debris_f1"], best["epoch"], rec["t_epoch_s"])
        torch.save({"epoch": epoch, "model": model.state_dict(), "ema": model_ema.ema.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                    "rng_numpy": np.random.get_state(), "rng_random": random.getstate(),
                    "rng_loader": g.get_state(), "best": best}, str(resume_file) + ".tmp")
        os.replace(str(resume_file) + ".tmp", resume_file)

    # ---- final evaluation of the selected checkpoint (same protocol as evaluate.py) ----
    if options["final_eval"]:
        from .evaluate import evaluate_split
        sel = build_student(options["width"], options["input_channels"], options["output_channels"]).to(device)
        sel.load_state_dict(torch.load(ckpt_dir / "best_ema.pth", map_location=device))
        sel.eval()
        results = {}
        for split in options["final_eval_splits"]:
            ds = make_dataset(options, split)
            cms = evaluate_split({"best": sel}, ds, ["single", "tta"], device)["best"]
            results[split] = {m: scores_from_confusion(cms[m], CLASS_NAMES) for m in cms}
            results[split]["n_patches"] = len(ds)
        write_json(rd / "metrics.json", {"run_id": options["run_id"], "width": options["width"], "seed": seed,
                                         "params": count_params(sel), "selected_epoch": best["epoch"],
                                         "selection": "best val macro-F1 of EMA weights", "results": results})
        for split, r in results.items():
            for m in ("single", "tta"):
                d = r[m]["per_class"][CLASS_NAMES[DEBRIS]]
                logging.info("FINAL %s %s: debris F1 %.1f R %.1f IoU %.1f | mIoU %.1f macroF1 %.1f", split, m,
                             100 * d["f1"], 100 * d["recall"], 100 * d["iou"], 100 * r[m]["mIoU"], 100 * r[m]["macro_f1"])
    logging.info("DONE")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path of the images")
    parser.add_argument("--cache", default="", help="verified .npy cache dir (src/debriskd/data.py)")
    # Official options (defaults unchanged)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--batch", default=5, type=int)
    parser.add_argument("--input_channels", default=11, type=int)
    parser.add_argument("--output_channels", default=15, type=int)
    parser.add_argument("--weight_param", default=1.03, type=float)
    parser.add_argument("--vscp", type=bool_flag, default=True)
    parser.add_argument("--label_smoothing", default=0.0, type=float)
    parser.add_argument("--clip_grad", default=None, type=float)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--decay", default=0, type=float)
    parser.add_argument("--lr_steps", default="[45,65]", type=str)
    parser.add_argument("--model_ema_decay", type=float, default=0.999)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--pin_memory", default=False, type=bool_flag)
    parser.add_argument("--prefetch_factor", default=2, type=int)
    parser.add_argument("--persistent_workers", default=False, type=bool_flag)
    # Ours
    parser.add_argument("--width", default=1.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--phase", default="p2")
    parser.add_argument("--setting", default="B")
    parser.add_argument("--run_id", default="", help="reuse to resume an existing run")
    parser.add_argument("--subset", default=0, type=int)
    parser.add_argument("--final_eval", default=True, type=bool_flag)
    parser.add_argument("--final_eval_splits", nargs="+", default=["val", "test"])
    # Phase 3: distillation (docs/PROJECT_BRIEF.md) and setting C
    parser.add_argument("--teacher_ckpt", default="", help="teacher state dict (official or ours); empty = no KD")
    parser.add_argument("--kd_T", default=2.0, type=float)
    parser.add_argument("--kd_tau", default=0.8, type=float)
    parser.add_argument("--lam_L", default=1.0, type=float)
    parser.add_argument("--lam_U", default=1.0, type=float)
    parser.add_argument("--debris_w", default=1.0, type=float)
    parser.add_argument("--debris_w_max", default=4.0, type=float)
    parser.add_argument("--kd_warmup_epochs", default=5, type=int)
    parser.add_argument("--kd_class_weighted", default=False, type=bool_flag,
                        help="weight KD terms by the official CE class weights (true class / teacher class)")
    parser.add_argument("--extra_debris_ce_weight", default=1.0, type=float, help="setting C; 1.0 = official")
    parser.add_argument("--config", default="", help="YAML of train.py options (configs/settings/*.yaml); CLI wins")
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        with open(pre.config) as f:
            parser.set_defaults(**(yaml.safe_load(f) or {}))
    args = parser.parse_args(argv)
    options = vars(args)
    lr_steps = ast.literal_eval(options["lr_steps"])
    options["lr_steps"] = [lr_steps] if isinstance(lr_steps, int) else list(lr_steps)
    if not options["run_id"]:
        options["run_id"] = make_run_id(options["phase"], options["setting"], f"w{options['width']}-s{options['seed']}")
    return options


if __name__ == "__main__":
    main(parse_args())
