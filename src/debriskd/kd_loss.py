"""Distillation loss: KD on labelled pixels + confidence-weighted KD on ignored pixels (docs/PROJECT_BRIEF.md).

    kl(x)  = T² · Σ_c p_t,c (log p_t,c − log p_s,c),   p = softmax(logits / T)
    c(x)   = max_c softmax(t)_c                         (teacher confidence at T = 1)
    w(x)   = clamp((c − τ) / (1 − τ), 0, 1)
    d(x)   = debris_w if argmax t == DEBRIS else 1, clipped at debris_w_max
    L      = L_sup + λ_L · mean_lab(kl) + λ_U(epoch) · mean_{ign, w>0}(w · d · kl)

Teacher outputs on ignored pixels are used only as soft targets, never as hard labels.
DEBRIS and IGNORE come from ``official.py`` (read from the dataset code in Phase 0: debris = 0, ignore = -1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .official import DEBRIS, IGNORE_INDEX


@dataclass
class KDConfig:
    T: float = 2.0
    tau: float = 0.8
    lam_L: float = 1.0
    lam_U: float = 1.0
    debris_w: float = 1.0
    debris_w_max: float = 4.0
    warmup_epochs: int = 5


def lam_U_at(epoch: int, cfg: KDConfig) -> float:
    """Linear warm-up of λ_U from 0 (epoch 1) to cfg.lam_U (reached at epoch warmup_epochs + 1)."""
    if cfg.warmup_epochs <= 0:
        return cfg.lam_U
    return cfg.lam_U * min(1.0, max(0.0, (epoch - 1) / cfg.warmup_epochs))


def kd_terms(s: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg: KDConfig,
             debris: int = DEBRIS, ignore: int = IGNORE_INDEX) -> Dict[str, torch.Tensor]:
    """Per-batch KD terms. ``s``, ``t``: logits [B, C, H, W] (t detached); ``y``: labels [B, H, W]."""
    t = t.detach()
    T = cfg.T
    log_pt = F.log_softmax(t / T, dim=1)
    log_ps = F.log_softmax(s / T, dim=1)
    kl = (log_pt.exp() * (log_pt - log_ps)).sum(1) * T ** 2            # [B, H, W]
    conf = F.softmax(t, dim=1).amax(1)                                  # T = 1
    w = ((conf - cfg.tau) / (1.0 - cfg.tau)).clamp(0, 1)
    d = torch.where(t.argmax(1) == debris, torch.full_like(w, cfg.debris_w), torch.ones_like(w))
    d = d.clamp(max=cfg.debris_w_max)
    lab = y != ignore
    ign = ~lab
    elig = ign & (w > 0)
    zero = s.sum() * 0.0                                                # keeps the graph, value 0
    L_lab = kl[lab].mean() if lab.any() else zero
    L_ign = (w * d * kl)[elig].mean() if elig.any() else zero
    with torch.no_grad():
        n_ign = int(ign.sum())
        frac_used = float(elig.sum()) / max(n_ign, 1)
        hist = torch.bincount(t.argmax(1)[elig], minlength=t.shape[1]) if elig.any() else torch.zeros(t.shape[1], dtype=torch.long, device=t.device)
    return {"kd_lab": L_lab, "kd_ign": L_ign, "frac_ign_used": torch.tensor(frac_used),
            "teacher_hist_ign": hist.cpu(), "kl": kl, "w": w}


def total_loss(L_sup: torch.Tensor, s: torch.Tensor, t: Optional[torch.Tensor], y: torch.Tensor,
               cfg: Optional[KDConfig], epoch: int) -> Dict[str, torch.Tensor]:
    """L = L_sup + λ_L·L_kd_lab + λ_U(epoch)·L_kd_ign. Without a teacher/config returns L_sup unchanged."""
    if t is None or cfg is None or (cfg.lam_L == 0 and cfg.lam_U == 0):
        return {"loss": L_sup, "sup": L_sup.detach()}
    k = kd_terms(s, t, y, cfg)
    lu = lam_U_at(epoch, cfg)
    loss = L_sup + cfg.lam_L * k["kd_lab"] + lu * k["kd_ign"]
    return {"loss": loss, "sup": L_sup.detach(), "kd_lab": k["kd_lab"].detach(), "kd_ign": k["kd_ign"].detach(),
            "lam_U": torch.tensor(lu), "frac_ign_used": k["frac_ign_used"], "teacher_hist_ign": k["teacher_hist_ign"]}
