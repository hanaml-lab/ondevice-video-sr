"""Distillation objective (Eq. 4) without the adversarial and LPIPS terms, which need external weights.

``perceptual`` is pluggable: pass an LPIPS module for full training; the toy setting uses a fixed
gradient-domain proxy so that it runs offline on CPU.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import warp


def temporal_residual(y: torch.Tensor, flow_hr: torch.Tensor | None) -> torch.Tensor:
    """r_t(y) = y_t - W(y_{t-1}, u_t) for a clip y of shape (B,T,C,H,W); returns (B,T-1,C,H,W)."""
    prev, cur = y[:, :-1], y[:, 1:]
    if flow_hr is None:
        return cur - prev
    b, t = prev.shape[:2]
    w = warp(prev.flatten(0, 1), flow_hr[:, 1:].flatten(0, 1)).view_as(prev)
    return cur - w


def consistency_matching(student: torch.Tensor, teacher: torch.Tensor, flow_hr: torch.Tensor | None = None):
    """|| r_t(student) - r_t(teacher) ||_1 : penalise flicker the teacher does not have."""
    return (temporal_residual(student, flow_hr) - temporal_residual(teacher, flow_hr)).abs().mean()


def residual_suppression(student: torch.Tensor, flow_hr: torch.Tensor | None = None):
    """The common alternative || r_t(student) ||_1 (admits temporal blur as a fixed point; Appendix B)."""
    return temporal_residual(student, flow_hr).abs().mean()


def gradient_proxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Offline stand-in for LPIPS: L1 between image gradients (emphasises texture, not just colour)."""
    def grads(x):
        return x[..., 1:, :] - x[..., :-1, :], x[..., :, 1:] - x[..., :, :-1]
    (ay, ax), (by, bx) = grads(a), grads(b)
    return (ay - by).abs().mean() + (ax - bx).abs().mean()


def distillation_loss(student, teacher, weights=(1.0, 1.0, 0.5), perceptual=gradient_proxy, flow_hr=None):
    l1 = F.l1_loss(student, teacher)
    lp = perceptual(student.flatten(0, 1), teacher.flatten(0, 1))
    cm = consistency_matching(student, teacher, flow_hr)
    total = weights[0] * l1 + weights[1] * lp + weights[2] * cm
    return total, dict(l1=l1.detach().item(), perceptual=lp.detach().item(), consistency=cm.detach().item())
