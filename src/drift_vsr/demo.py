"""DRIFT on the CPU toy world: train the student with and without the contraction penalty and compare.

    python -m drift_vsr.demo [--config configs/toy.yaml] [--save student.pt]

Both runs do an FP warm-up and then long-rollout QAT with the state bank; they differ only in the penalty on the
state-Jacobian norm. Each is evaluated in W8A8 fake quantisation on a held-out 120-frame clip of the moving-texture
world, whose high-resolution frames stand in for the teacher's outputs.
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


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/toy.yaml")
    ap.add_argument("--save", default=None, help="write the DRIFT student's weights and config here (for examples/)")
    args = ap.parse_args(argv)
    cfg = Config(**yaml.safe_load(config_path(args.config).read_text()))
    t0 = time.time()
    runs = {"DRIFT (contraction penalty)": train(cfg, log=lambda *_: None),
            "same, without the penalty": train(replace(cfg, lambda_contr=0.0), log=lambda *_: None)}
    print(f"W8A8 student, {sum(p.numel() for p in runs['same, without the penalty']['model'].parameters()):,} parameters; "
          f"{cfg.iters_fp} FP + {cfg.iters_qat} QAT steps; held-out clip of {cfg.eval_frames} frames, x{cfg.scale}")
    print(f"{'':<29}{'PSNR':>7}{'bilinear':>9}{'drift':>7}{'sigma mean':>11}{'sigma max':>10}{'err first':>10}{'err last':>9}")
    for name, r in runs.items():
        print(f"{name:<29}{r['psnr']:>7.2f}{r['psnr_bilinear']:>9.2f}{r['drift_psnr']:>7.2f}{r['sigma_mean']:>11.2f}"
              f"{r['sigma_max']:>10.2f}{r['state_error_rel_first']:>10.3f}{r['state_error_rel_final']:>9.3f}")
    print(f"target gamma = {cfg.gamma}; drift: PSNR of the first 10 frames minus the last 10 (dB); err: "
          f"||h_quant - h_float|| / ||h_float|| at the first and last frame")
    print(f"done in {time.time() - t0:.1f}s")
    if args.save:
        drift = runs["DRIFT (contraction penalty)"]
        torch.save({"config": asdict(cfg), "state_dict": drift["model"].state_dict()}, args.save)
    return runs


if __name__ == "__main__":
    main()
