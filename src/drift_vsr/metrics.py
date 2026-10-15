"""Trajectory-level evaluation: drift@T, state error under quantisation, and warping error."""
from __future__ import annotations

from typing import Callable, Sequence

import torch

from .quant import set_quant


def drift_at(scores: Sequence[float], T: int, k: int = 10) -> float:
    """drift@T = mean score of frames 1..k minus mean score of frames T-k+1..T (higher = worse)."""
    if T > len(scores) or T < k:
        raise ValueError(f"need k <= T <= len(scores); got T={T}, k={k}, len={len(scores)}")
    head = sum(scores[:k]) / k
    tail = sum(scores[T - k:T]) / k
    return float(head - tail)


@torch.no_grad()
def state_error_trajectory(model, stream: torch.Tensor) -> list[float]:
    """||h~_t - h_t|| per frame: run the same stream in float and in fake-quant mode (Proposition 1)."""
    out = []
    set_quant(model, False)
    hf = model.init_state(stream[:, 0])
    hq = hf.clone()
    xp = stream[:, 0]
    for t in range(stream.shape[1]):
        x = stream[:, t]
        set_quant(model, False)
        hf, _, _ = model.step(hf, xp, x)
        set_quant(model, True)
        hq, _, _ = model.step(hq, xp, x)
        out.append(float((hq - hf).flatten(1).norm(dim=1).mean()))
        xp = x
    set_quant(model, True)
    return out


def relative_state_error(model, stream: torch.Tensor) -> list[float]:
    """||h~_t - h_t|| / ||h_t|| per frame: the quantised state's error relative to the size of the float state."""
    out = []
    set_quant(model, False)
    hf = model.init_state(stream[:, 0])
    hq = hf.clone()
    xp = stream[:, 0]
    with torch.no_grad():
        for t in range(stream.shape[1]):
            x = stream[:, t]
            set_quant(model, False)
            hf, _, _ = model.step(hf, xp, x)
            set_quant(model, True)
            hq, _, _ = model.step(hq, xp, x)
            err = (hq - hf).flatten(1).norm(dim=1) / hf.flatten(1).norm(dim=1).clamp_min(1e-12)
            out.append(float(err.mean()))
            xp = x
    set_quant(model, True)
    return out


def state_jacobian_norms(model, stream: torch.Tensor, iters: int = 8, stride: int = 1) -> list[float]:
    """Power-iteration estimate of ||d h_t / d h_{t-1}||_2 of the real-valued transition along the float trajectory
    (Phi_t in Proposition 1), at every ``stride``-th frame."""
    from .contraction import jacobian_spectral_norm

    out = []
    set_quant(model, False)
    with torch.no_grad():
        h = model.init_state(stream[:, 0])
    xp = stream[:, 0]
    for t in range(stream.shape[1]):
        x = stream[:, t]
        if t % stride == 0:
            sigma, _ = jacobian_spectral_norm(lambda s, xp=xp, x=x: model.step(s, xp, x)[0], h, iters=iters)
            out.append(float(sigma.detach().mean()))
        with torch.no_grad():
            h, _, _ = model.step(h, xp, x)
        xp = x
    set_quant(model, True)
    return out


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.clamp(0, 1) - b.clamp(0, 1)).pow(2).mean()
    return float(10 * torch.log10(1.0 / mse.clamp_min(1e-12)))


def per_frame(fn: Callable[[torch.Tensor, torch.Tensor], float], pred: torch.Tensor, ref: torch.Tensor) -> list[float]:
    return [fn(pred[:, t], ref[:, t]) for t in range(pred.shape[1])]
