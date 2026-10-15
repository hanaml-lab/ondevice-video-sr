import math

import pytest
import torch

from drift_vsr import (LSQ, DriftStudent, StateBank, consistency_matching, drift_at, drift_bound,
                       jacobian_spectral_norm, residual_suppression, set_quant, warp)
from drift_vsr.train import Config, train


def test_lsq_levels_and_ste():
    q = LSQ(bits=4, signed=True)
    x = torch.linspace(-1, 1, 101, requires_grad=True)
    y = q(x)
    levels = torch.unique((y / q.step).detach().round())
    assert len(levels) <= 16
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_spectral_norm_linear_map():
    torch.manual_seed(0)
    A = torch.randn(8, 8)
    true = torch.linalg.matrix_norm(A, 2)
    fn = lambda h: h @ A.T  # noqa: E731
    v = None
    for _ in range(30):
        s, v = jacobian_spectral_norm(fn, torch.zeros(1, 8), iters=2, v0=v)
    assert float(s) == pytest.approx(float(true), rel=1e-3)
    assert float(s) <= float(true) * (1 + 1e-4)


def test_drift_bound():
    assert drift_bound(0.1, 0.9) == pytest.approx(1.0)
    assert drift_bound(0.1, 1.0, t=50) == pytest.approx(5.0)
    assert math.isinf(drift_bound(0.1, 1.2))
    # simulated scalar contraction never exceeds the bound
    g, e, d = 0.9, 0.1, 0.0
    for t in range(1, 500):
        d = g * d + e
        assert d <= drift_bound(e, g, t=t) + 1e-9


def test_drift_at():
    s = [1.0] * 10 + [0.5] * 90
    assert drift_at(s, 100) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        drift_at(s, 200)


def test_consistency_matching_vs_suppression():
    t = torch.arange(8).float()
    teacher = (0.5 + 0.2 * torch.sin(t))[None, :, None, None, None].expand(1, 8, 3, 4, 4)
    assert float(consistency_matching(teacher, teacher)) == 0.0
    assert float(residual_suppression(teacher)) > 0.0


def test_warp_identity():
    x = torch.rand(2, 3, 8, 8)
    assert torch.allclose(warp(x, torch.zeros(2, 2, 8, 8)), x, atol=1e-5)


def test_statebank():
    b = StateBank(horizon=30, p_bank=1.0)
    b.store(0, torch.ones(1, 2), 15)
    assert b.fetch(0).offset == 15
    b.store(0, torch.ones(1, 2), 45)
    assert b.fetch(0) is None


def test_model_shapes_and_quant_toggle():
    m = DriftStudent(width=8, blocks_f=1, blocks_g=1)
    clip = torch.rand(2, 3, 3, 8, 8)
    out, h, ctx = m(clip)
    assert out.shape == (2, 3, 3, 32, 32) and h.shape == (2, 8, 8, 8) and len(ctx) == 3
    set_quant(m, False)
    out2, _, _ = m(clip)
    assert out2.shape == out.shape
    assert 0 < float(m.alpha.detach()) < 1


def test_train_end_to_end_tiny():
    cfg = Config(iters_fp=4, iters_qat=4, unroll=3, videos=2, video_frames=12, bank_horizon=12, eval_frames=20)
    res = train(cfg, log=lambda s: None)
    assert math.isfinite(res["psnr"]) and len(res["history"]) == 8
    assert any(h["sigma"] > 0 for h in res["history"] if h["qat"])
