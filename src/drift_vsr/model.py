"""Unidirectional recurrent student with a gated contractive state update (Eq. 1-2 of the paper)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import LSQ, QConv2d


def warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp ``x`` (B,C,H,W) with ``flow`` (B,2,H,W) in pixels, bilinear, border padding."""
    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    gx = (xx[None] + flow[:, 0]) / max(w - 1, 1) * 2 - 1
    gy = (yy[None] + flow[:, 1]) / max(h - 1, 1) * 2 - 1
    return F.grid_sample(x, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border", align_corners=True)


class Block(nn.Module):
    def __init__(self, c: int, q: dict):
        super().__init__()
        self.c1 = QConv2d(c, c, 3, padding=1, **q)
        self.c2 = QConv2d(c, c, 3, padding=1, **q)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(x)))


class DriftStudent(nn.Module):
    """h_t = (1-g) * alpha * W(h_{t-1}) + g * c ;  y_t = G(h_t, x_t) + up(x_t)."""

    def __init__(self, width: int = 48, blocks_f: int = 8, blocks_g: int = 4, scale: int = 4,
                 bits_w: int = 8, bits_a: int = 8):
        super().__init__()
        q = dict(bits_w=bits_w, bits_a=bits_a)
        self.width, self.scale = width, scale
        self.flow = nn.Sequential(QConv2d(6, 16, 3, padding=1, **q), nn.ReLU(), QConv2d(16, 2, 3, padding=1, **q))
        self.inp = QConv2d(3 + width, width, 3, padding=1, **q)
        self.body = nn.Sequential(*[Block(width, q) for _ in range(blocks_f)])
        self.gate = QConv2d(width, width, 3, padding=1, **q)
        self.cand = QConv2d(width, width, 3, padding=1, **q)
        self.alpha_logit = nn.Parameter(torch.tensor(2.0))
        self.head = nn.Sequential(*[Block(width, q) for _ in range(blocks_g)],
                                  QConv2d(width, 3 * scale * scale, 3, padding=1, **q))
        self.state_q = LSQ(bits_a, signed=True)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def init_state(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.shape[0], self.width, x.shape[2], x.shape[3])

    def motion(self, x_prev: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.flow(torch.cat([x, x_prev], 1))

    def transition(self, h_prev: torch.Tensor, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Phi_t(h): the state map whose Lipschitz modulus in h the contraction regulariser controls."""
        hb = warp(h_prev, flow)
        z = self.body(self.inp(torch.cat([x, hb], 1)))
        g = torch.sigmoid(self.gate(z))
        c = torch.tanh(self.cand(z))
        return (1 - g) * self.alpha * hb + g * c

    def step(self, h_prev, x_prev, x):
        flow = self.motion(x_prev, x)
        h = self.state_q(self.transition(h_prev, x, flow))
        y = F.pixel_shuffle(self.head(h), self.scale)
        y = y + F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return h, y, flow

    def forward(self, clip: torch.Tensor, h0: torch.Tensor | None = None):
        """clip: (B,T,3,H,W). Returns (outputs (B,T,3,sH,sW), final state, per-step (h_prev, x, flow))."""
        h = self.init_state(clip[:, 0]) if h0 is None else h0
        outs, ctx = [], []
        x_prev = clip[:, 0]
        for t in range(clip.shape[1]):
            x = clip[:, t]
            hp = h
            h, y, flow = self.step(h, x_prev, x)
            outs.append(y)
            ctx.append((hp, x, flow))
            x_prev = x
        return torch.stack(outs, 1), h, ctx
