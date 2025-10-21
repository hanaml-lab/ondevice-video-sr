"""Contraction regulariser (Eq. 3): penalise (||d Phi / d h||_2 - gamma)_+^2 via power iteration.

The spectral norm of the state Jacobian is estimated with Jacobian-vector and vector-Jacobian products,
so the Jacobian itself is never formed. Power iteration under-estimates the norm, hence the penalty is a
lower bound on the true violation; warm-starting ``v`` across steps tightens it.
"""
from __future__ import annotations

from typing import Callable

import torch


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, *[1] * (v.dim() - 1))


def _jvp_fd(fn, h, v, eps):
    """Central finite-difference Jacobian-vector product; differentiable w.r.t. the parameters of fn.

    Forward-mode AD is unavailable for bilinear grid sampling in PyTorch, and the warp is part of Phi_t.
    """
    return (fn(h + eps * v) - fn(h - eps * v)) / (2 * eps)


def jacobian_spectral_norm(fn: Callable[[torch.Tensor], torch.Tensor], h: torch.Tensor, iters: int = 2,
                           v0: torch.Tensor | None = None, eps: float = 1e-3):
    """Return (sigma_hat, v) with sigma_hat ~ ||J||_2 at h (per batch element), J = d fn / d h.

    ``iters - 1`` power-iteration steps v <- J^T J v / ||.|| use a finite-difference JVP and an autograd VJP
    without building a parameter graph; the final ||J v|| keeps the graph so the penalty can be minimised.
    """
    h = h.detach()
    v = _unit(torch.randn_like(h) if v0 is None else v0.detach().clone())
    for _ in range(max(iters - 1, 0)):
        with torch.no_grad():
            jv = _jvp_fd(fn, h, v, eps)
        hr = h.clone().requires_grad_(True)
        with torch.enable_grad():
            (w,) = torch.autograd.grad(fn(hr), hr, jv)
        v = _unit(w.detach())
    jv = _jvp_fd(fn, h, v, eps)
    return jv.flatten(1).norm(dim=1), v


def contraction_penalty(sigma: torch.Tensor, gamma: float) -> torch.Tensor:
    return torch.relu(sigma - gamma).pow(2).mean()


def drift_bound(eps: float, gamma: float, delta0: float = 0.0, t: int | None = None) -> float:
    """Proposition 1: ||delta_t|| <= gamma^t delta0 + eps (1-gamma^t)/(1-gamma); t=None gives the sup."""
    if gamma >= 1.0:
        if t is None:
            return float("inf")
        return gamma**t * delta0 + (eps * t if gamma == 1.0 else eps * (gamma**t - 1) / (gamma - 1))
    if t is None:
        return delta0 + eps / (1 - gamma)
    return gamma**t * delta0 + eps * (1 - gamma**t) / (1 - gamma)
