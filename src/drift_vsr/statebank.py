"""State bank for long-rollout QAT (Section 3.3, Algorithm 1).

Stores detached quantised states per training video together with the frame offset they correspond to,
so a 15-frame unroll can resume from frame ``tau`` of a much longer rollout.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class Entry:
    state: torch.Tensor
    offset: int


class StateBank:
    def __init__(self, horizon: int = 300, p_bank: float = 0.7, seed: int = 0):
        self.horizon, self.p_bank = horizon, p_bank
        self._d: dict[int, Entry] = {}
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._d)

    def fetch(self, video_id: int) -> Entry | None:
        """Return a stored state with probability p_bank (if one exists), else None (start from zero)."""
        e = self._d.get(video_id)
        if e is None or self._rng.random() >= self.p_bank:
            return None
        return e

    def store(self, video_id: int, state: torch.Tensor, offset: int) -> None:
        if offset > self.horizon:
            self._d.pop(video_id, None)
            return
        self._d[video_id] = Entry(state.detach().clone(), offset)
