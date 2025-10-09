"""Learned step-size quantisation (LSQ, Esser et al. 2020) with a straight-through estimator."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    return (x - x * scale).detach() + x * scale


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


class LSQ(nn.Module):
    """Fake quantiser. ``per_channel`` quantises along dim 0 (weights); otherwise per tensor."""

    def __init__(self, bits: int = 8, signed: bool = True, per_channel: int = 0):
        super().__init__()
        self.bits, self.signed = bits, signed
        self.qn = -(2 ** (bits - 1)) if signed else 0
        self.qp = 2 ** (bits - 1) - 1 if signed else 2**bits - 1
        self.step = nn.Parameter(torch.ones(max(per_channel, 1)))
        self.per_channel = per_channel
        self.enabled = True
        self.register_buffer("initialised", torch.tensor(False))

    @torch.no_grad()
    def _init(self, x: torch.Tensor) -> None:
        if self.per_channel:
            m = x.detach().abs().flatten(1).quantile(0.9999, dim=1)
        else:
            m = x.detach().abs().flatten().quantile(0.9999).reshape(1)
        self.step.copy_((m / self.qp).clamp_min(1e-8))
        self.initialised.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if not bool(self.initialised):
            self._init(x)
        g = 1.0 / math.sqrt(x.numel() / max(self.per_channel, 1) * self.qp)
        s = _grad_scale(self.step, g)
        if self.per_channel:
            s = s.view(-1, *([1] * (x.dim() - 1)))
        return _round_ste((x / s).clamp(self.qn, self.qp)) * s


class QConv2d(nn.Conv2d):
    """Conv2d with LSQ on weights (per channel) and on its input activation (per tensor)."""

    def __init__(self, *args, bits_w: int = 8, bits_a: int = 8, **kw):
        super().__init__(*args, **kw)
        self.wq = LSQ(bits_w, signed=True, per_channel=self.out_channels)
        self.aq = LSQ(bits_a, signed=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(self.aq(x), self.wq(self.weight), self.bias, self.stride, self.padding)


def set_quant(module: nn.Module, enabled: bool) -> None:
    for m in module.modules():
        if isinstance(m, LSQ):
            m.enabled = enabled
