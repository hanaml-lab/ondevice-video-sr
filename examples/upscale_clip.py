"""Upscale a clip of your own with a DRIFT student and measure how its integer state drifts.

    python -m drift_vsr.demo --save student.pt          # trains the toy student and saves it
    python examples/upscale_clip.py examples/data/clip_lr.npy --checkpoint student.pt \
        --reference examples/data/clip_hr.npy --out upscaled.npy

The clip is a NumPy array of frames, shape (T, H, W, 3), uint8 or floats in [0, 1]. The student runs in W8A8 fake
quantisation, as on the device. The script reports the state-Jacobian norm estimate along the clip (Proposition 1
bounds the accumulated error when it stays below 1), the quantised state's error relative to the float state at the
first and last frames, and, given a reference at the output resolution, PSNR against bilinear upsampling and the
drift between the first and the last ten frames.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from drift_vsr import DriftStudent, drift_at, set_quant
from drift_vsr.metrics import per_frame, psnr, relative_state_error, state_jacobian_norms


def load_clip(path: str) -> torch.Tensor:
    a = np.load(path)
    a = a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32)
    return torch.from_numpy(a).permute(0, 3, 1, 2)[None]  # (1, T, 3, H, W)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clip", help=".npy of low-resolution frames, (T, H, W, 3)")
    ap.add_argument("--checkpoint", required=True, help="a student saved by python -m drift_vsr.demo --save")
    ap.add_argument("--reference", help=".npy of reference frames at the output resolution, for PSNR")
    ap.add_argument("--out", help="write the upscaled frames here as uint8 (T, H, W, 3)")
    args = ap.parse_args(argv)
    ck = torch.load(args.checkpoint)
    c = ck["config"]
    if c.get("threads"):
        torch.set_num_threads(c["threads"])  # the CPU threads the student was trained with
    model = DriftStudent(c["width"], c["blocks_f"], c["blocks_g"], c["scale"], c["bits_w"], c["bits_a"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    lr = load_clip(args.clip)
    with torch.no_grad():
        set_quant(model, True)
        out, _, _ = model(lr)
    sig = state_jacobian_norms(model, lr, stride=2)
    rel = relative_state_error(model, lr)
    T = lr.shape[1]
    print(f"{T} frames, {lr.shape[-2]}x{lr.shape[-1]} -> {out.shape[-2]}x{out.shape[-1]}, W8A8 student "
          f"({sum(p.numel() for p in model.parameters()):,} parameters)")
    print(f"state-Jacobian norm estimate: mean {np.mean(sig):.2f}, max {np.max(sig):.2f} (below 1 bounds the drift)")
    print(f"quantised state error relative to float: {rel[0]:.3f} at frame 1, {rel[-1]:.3f} at frame {T}")
    result = {"sigma": sig, "relative_state_error": rel}
    if args.reference:
        hr = load_clip(args.reference)
        scores = per_frame(psnr, out, hr)
        bil = F.interpolate(lr.flatten(0, 1), scale_factor=c["scale"], mode="bilinear", align_corners=False).view_as(hr)
        base = per_frame(psnr, bil, hr)
        print(f"PSNR {np.mean(scores):.2f} dB against {np.mean(base):.2f} dB for bilinear; "
              f"drift@{T} {drift_at(scores, T):.2f} dB (first 10 frames minus last 10)")
        result.update(psnr=float(np.mean(scores)), psnr_bilinear=float(np.mean(base)))
    if args.out:
        frames = (out[0].clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
        np.save(args.out, frames)
        print(f"wrote {args.out}")
    return result


if __name__ == "__main__":
    main()
