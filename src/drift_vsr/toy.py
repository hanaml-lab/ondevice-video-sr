"""A synthetic moving-texture video world small enough for CPU.

High-resolution frames are a band-limited random texture translated with a slowly varying velocity;
low-resolution inputs are blurred, 4x average-pooled and noised. The HR frames play the role of the
teacher's outputs, so the full DRIFT objective can be exercised end to end in seconds.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _texture(g: torch.Generator, size: int, channels: int = 3) -> torch.Tensor:
    z = torch.randn(1, channels, size, size, generator=g)
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
    k = (k[:, None] * k[None]) / 256
    k = k.expand(channels, 1, 5, 5)
    for _ in range(6):
        z = F.conv2d(F.pad(z, (2, 2, 2, 2), mode="circular"), k, groups=channels)
    z = (z - z.mean()) / z.std()
    return (0.5 + 0.18 * z).clamp(0, 1)


def make_clip(frames: int, hr: int = 64, scale: int = 4, noise: float = 0.01, seed: int = 0):
    """Return (lr, hr) clips of shape (1,T,3,hr/scale,hr/scale) and (1,T,3,hr,hr)."""
    g = torch.Generator().manual_seed(seed)
    tex = _texture(g, hr * 2)
    vx, vy = (torch.rand(2, generator=g) * 2 - 1).tolist()
    his, los = [], []
    px = py = 0.0
    for t in range(frames):
        px += vx + 0.3 * torch.randn(1, generator=g).item()
        py += vy + 0.3 * torch.randn(1, generator=g).item()
        shift = torch.roll(tex, shifts=(int(round(py)) % (2 * hr), int(round(px)) % (2 * hr)), dims=(2, 3))
        y = shift[..., :hr, :hr]
        x = F.avg_pool2d(y, scale)
        x = (x + noise * torch.randn(x.shape, generator=g)).clamp(0, 1)
        his.append(y)
        los.append(x)
    return torch.stack(los, 1), torch.stack(his, 1)


def make_dataset(n_videos: int, frames: int, hr: int = 64, scale: int = 4, seed: int = 0):
    return [make_clip(frames, hr, scale, seed=seed * 1000 + i) for i in range(n_videos)]
