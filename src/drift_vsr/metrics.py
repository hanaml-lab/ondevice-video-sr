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


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.clamp(0, 1) - b.clamp(0, 1)).pow(2).mean()
    return float(10 * torch.log10(1.0 / mse.clamp_min(1e-12)))


def per_frame(fn: Callable[[torch.Tensor, torch.Tensor], float], pred: torch.Tensor, ref: torch.Tensor) -> list[float]:
    return [fn(pred[:, t], ref[:, t]) for t in range(pred.shape[1])]
