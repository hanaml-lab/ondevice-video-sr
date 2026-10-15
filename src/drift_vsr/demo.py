"""DRIFT on the CPU toy world: train the student with and without the contraction penalty and compare.

    python -m drift_vsr.demo [--config configs/toy.yaml] [--save student.pt]

Both runs do an FP warm-up and then long-rollout QAT with the state bank; they differ only in the penalty on the
state-Jacobian norm. Each is evaluated in W8A8 fake quantisation on a held-out stream of the moving-texture world, whose
high-resolution frames stand in for the teacher's outputs, and Proposition 1 is measured along that stream: the
error the rounded recurrent state accumulates against the real-valued one, the largest error one step injects (eps)
and the largest factor by which one step carries the accumulated error forward (gamma*).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import yaml

from .train import Config, train


def config_path(name: str) -> Path:
    """``name`` from the working directory, else from the checkout this package is installed from."""
    path = Path(name)
    if not path.exists() and (Path(__file__).resolve().parents[2] / name).exists():
        return Path(__file__).resolve().parents[2] / name
    return path


def windows(T: int) -> list[tuple[int, int]]:
    """Frame windows (1-based, inclusive) at which the accumulated error is reported: early, 100, 1000 and the end."""
    out = []
    for a, b in ((1, 10), (91, 100), (991, 1000), (T - 9, T)):
        if 1 <= a and b <= T and (a, b) not in out:
            out.append((a, b))
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/toy.yaml")
    ap.add_argument("--save", default=None, help="write the DRIFT student's weights and config here (for examples/)")
    args = ap.parse_args(argv)
    cfg = Config(**yaml.safe_load(config_path(args.config).read_text()))
    t0 = time.time()
    runs = {"DRIFT": train(cfg, log=lambda *_: None),
            "no penalty": train(replace(cfg, lambda_contr=0.0), log=lambda *_: None)}
    T = cfg.eval_frames
    print(f"W8A8 student, {sum(p.numel() for p in runs['no penalty']['model'].parameters()):,} parameters; "
          f"{cfg.iters_fp} FP + {cfg.iters_qat} QAT steps on {cfg.video_frames}-frame videos; "
          f"held-out stream of {T:,} frames, x{cfg.scale}")
    print(f"{'':<30}{'PSNR':>6}{'bilinear':>10}{'drift':>7}{'sigma mean':>12}{'sigma max':>11}")
    for name, label in (("DRIFT", "DRIFT (contraction penalty)"), ("no penalty", "same, without the penalty")):
        r = runs[name]
        print(f"{label:<30}{r['psnr']:>6.2f}{r['psnr_bilinear']:>10.2f}{r['drift_psnr']:>7.2f}{r['sigma_mean']:>12.2f}"
              f"{r['sigma_max']:>11.2f}")
    print()
    print("Accumulated state error ||h_round - h_float||, in units of eps, mean over frames:")
    wins = windows(T)
    print(f"{'':<30}{'eps':>7}{'gamma*':>8}{'1/(1-gamma*)':>14}" + "".join(f"{f'{a}-{b}':>{max(8, len(f'{a}-{b}') + 2)}}" for a, b in wins) + f"{'max':>7}")
    for name in ("DRIFT", "no penalty"):
        for mode, what in (("state", "state rounded only"), ("w8a8", "W8A8")):
            d = runs[name]["rounding"][mode]
            eps, g = d["eps"], d["gamma"]
            lim = f"{1 / (1 - g):.1f}" if g < 1 else "-"
            cells = "".join(f"{sum(d['err'][a - 1:b]) / (b - a + 1) / eps:>{max(8, len(f'{a}-{b}') + 2)}.2f}" for a, b in wins)
            print(f"{name + ', ' + what:<30}{eps:>7.4f}{g:>8.2f}{lim:>14}{cells}{max(d['err']) / eps:>7.2f}")
    print(f"eps: the largest error one step injects; gamma*: the largest factor by which one step carries the error "
          f"forward. Proposition 1 bounds the error by eps/(1-gamma*) when gamma* < 1 (target gamma = {cfg.gamma}).")
    print(f"drift: PSNR of the first 10 frames minus the last 10 (dB); sigma: state-Jacobian norm estimate at every "
          f"{cfg.jacobian_stride}th frame")
    print(f"done in {time.time() - t0:.1f}s")
    if args.save:
        drift = runs["DRIFT"]
        torch.save({"config": asdict(cfg), "state_dict": drift["model"].state_dict()}, args.save)
    return runs


if __name__ == "__main__":
    main()
