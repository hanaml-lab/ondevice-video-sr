"""DRIFT training loop (Algorithm 1): long-rollout QAT with a state bank and a contraction penalty."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field

import torch
import yaml

from .contraction import contraction_penalty, jacobian_spectral_norm
from .losses import distillation_loss
from .metrics import drift_at, per_frame, psnr, rounding_decomposition, state_jacobian_norms
from .model import DriftStudent
from .quant import set_quant
from .statebank import StateBank
from .toy import make_clip, make_dataset


@dataclass
class Config:
    width: int = 16
    blocks_f: int = 2
    blocks_g: int = 1
    scale: int = 4
    bits_w: int = 8
    bits_a: int = 8
    hr: int = 32
    videos: int = 4
    video_frames: int = 60
    unroll: int = 6
    iters_fp: int = 150
    iters_qat: int = 150
    lr: float = 2e-3
    gamma: float = 0.9
    lambda_contr: float = 0.1
    power_iters: int = 2
    bank_horizon: int = 60
    p_bank: float = 0.7
    loss_weights: list = field(default_factory=lambda: [1.0, 1.0, 0.5])
    eval_frames: int = 120
    jacobian_stride: int = 4  # estimate the state-Jacobian norm at every n-th frame of the evaluation stream
    seed: int = 0
    threads: int = 0  # torch CPU threads; 0 keeps torch's default

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            return cls(**(yaml.safe_load(f) or {}))


def train(cfg: Config, log=print) -> dict:
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    model = DriftStudent(cfg.width, cfg.blocks_f, cfg.blocks_g, cfg.scale, cfg.bits_w, cfg.bits_a)
    data = make_dataset(cfg.videos, cfg.video_frames, cfg.hr, cfg.scale, seed=cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.99))
    bank = StateBank(cfg.bank_horizon, cfg.p_bank, cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    v_warm: dict[int, torch.Tensor] = {}
    hist = []
    t0 = time.time()
    total = cfg.iters_fp + cfg.iters_qat
    for it in range(total):
        qat = it >= cfg.iters_fp
        set_quant(model, qat)
        vid = int(torch.randint(len(data), (1,), generator=gen))
        lr_clip, hr_clip = data[vid]
        entry = bank.fetch(vid) if qat else None
        tau = entry.offset if entry is not None else 0
        tau = min(tau, cfg.video_frames - cfg.unroll)
        h0 = entry.state if entry is not None else None
        x = lr_clip[:, tau:tau + cfg.unroll]
        y = hr_clip[:, tau:tau + cfg.unroll]
        out, h_last, ctx = model(x, h0)
        loss, parts = distillation_loss(out, y, cfg.loss_weights)
        sigma = torch.zeros(1)
        if qat and cfg.lambda_contr > 0:
            hp, xt, flow = ctx[-1]
            fn = lambda h: model.transition(h, xt.detach(), flow.detach())  # noqa: E731
            # Phi_t is the real-valued map; rounding enters Proposition 1 as the additive eta_t.
            set_quant(model, False)
            sigma, v_warm[vid] = jacobian_spectral_norm(fn, hp.detach(), cfg.power_iters, v_warm.get(vid))
            set_quant(model, True)
            loss = loss + cfg.lambda_contr * contraction_penalty(sigma, cfg.gamma)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if qat:
            bank.store(vid, h_last, tau + cfg.unroll)
        hist.append(dict(it=it, loss=loss.detach().item(), sigma=sigma.detach().mean().item(), qat=qat, **parts))
        if it % 20 == 0 or it == total - 1:
            log(f"it {it:4d} {'QAT' if qat else 'FP '} loss {loss.detach().item():.4f} sigma {sigma.detach().mean().item():.3f}")
    res = evaluate(model, cfg)
    res.update(seconds=time.time() - t0, history=hist, config=asdict(cfg), model=model)
    return res


@torch.no_grad()
def evaluate(model: DriftStudent, cfg: Config) -> dict:
    """W8A8 quality on a held-out stream, and Proposition 1 measured along it.

    For the recurrent state rounded alone ("state") and for the deployed W8A8 student ("w8a8"), ``rounding[mode]``
    holds eps, the largest error a single step injects; gamma, the largest one-step gain of the accumulated error;
    the bound eps / (1 - gamma) (infinite unless gamma < 1); and, per frame, the accumulated error ``err``.
    """
    lr, hr = make_clip(cfg.eval_frames, cfg.hr, cfg.scale, seed=10_000 + cfg.seed)
    dec = rounding_decomposition(model, lr, ("state", "w8a8"), ref=hr)
    scores = dec["w8a8"].pop("psnr")
    dec["state"].pop("psnr")
    bic = torch.nn.functional.interpolate(lr.flatten(0, 1), scale_factor=cfg.scale, mode="bilinear",
                                          align_corners=False).view_as(hr)
    rounding = {}
    for mode, d in dec.items():
        eps = max(d["inj"])
        gamma = max((g for g in d["gain"] if g is not None), default=0.0)
        rounding[mode] = dict(eps=eps, gamma=gamma, bound=eps / (1 - gamma) if gamma < 1 else float("inf"), **d)
    with torch.enable_grad():
        sig = state_jacobian_norms(model, lr, stride=cfg.jacobian_stride)
    w = rounding["w8a8"]
    rel = [e / max(n, 1e-12) for e, n in zip(w["err"], w["norm"])]
    return dict(psnr=sum(scores) / len(scores), psnr_bilinear=sum(per_frame(psnr, bic, hr)) / len(scores),
                drift_psnr=drift_at(scores, cfg.eval_frames), state_error_final=w["err"][-1],
                state_error_max=max(w["err"]), state_error_rel_first=rel[0], state_error_rel_final=rel[-1],
                state_error_rel_max=max(rel), sigma_mean=sum(sig) / len(sig), sigma_max=max(sig),
                alpha=float(model.alpha), rounding=rounding)


def main(argv=None):
    p = argparse.ArgumentParser(description="Train a DRIFT student on the CPU toy world.")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=None, help="write metrics JSON here")
    p.add_argument("--no-contraction", action="store_true")
    a = p.parse_args(argv)
    cfg = Config.from_yaml(a.config) if a.config else Config()
    if a.no_contraction:
        cfg.lambda_contr = 0.0
    res = train(cfg)
    summary = {k: v for k, v in res.items() if k not in ("history", "model", "rounding")}
    summary["rounding"] = {m: {k: r[k] for k in ("eps", "gamma", "bound")} for m, r in res["rounding"].items()}
    print(json.dumps(summary, indent=1, default=str))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({k: v for k, v in res.items() if k != "model"}, f, indent=1, default=str)
    return res


if __name__ == "__main__":
    main()
