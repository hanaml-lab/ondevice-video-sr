# DRIFT: drift-bounded distillation for on-device video super-resolution

[![ci](https://github.com/hanaml-lab/ondevice-video-sr/actions/workflows/ci.yml/badge.svg)](https://github.com/hanaml-lab/ondevice-video-sr/actions/workflows/ci.yml)

Reference implementation of *DRIFT: Drift-Bounded Distillation of One-Step Video Diffusion into Integer Recurrent Students for Real-Time On-Device Super-Resolution* (Agarwal, hanaML technical report, 2025).

When a recurrent video super-resolution network runs in INT8, rounding error enters the hidden state every frame
and builds up over the stream. DRIFT distils a one-step video diffusion teacher (DOVE) into a 1.9M-parameter
unidirectional recurrent student and bounds that error by construction:

- **Gated contractive recurrence.** `h_t = (1-g)·α·W(h_{t-1}) + g·c`. A contraction regulariser pushes the
  state-Jacobian spectral norm below γ, which bounds the accumulated error by `ε/(1-γ)` at every horizon
  (Proposition 1). The norm is estimated by power iteration and the Jacobian is never formed.
- **Long-rollout QAT.** LSQ fake quantisation on weights, activations and the state. A state bank lets a
  15-frame unroll resume from frame τ ≤ 300 of an earlier rollout.
- **Consistency matching.** The student matches the teacher's flow-compensated temporal residual
  `r_t(ŷ) ≈ r_t(y^T)`. It does not simply suppress temporal change.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch, if you have no GPU
pip install -e ".[dev]"          # torch, numpy, pyyaml; pytest and ruff
pip install -e ".[paper]"        # + matplotlib, to regenerate the report's figures
```

## Quickstart (CPU, about a minute)

```bash
python -m drift_vsr.demo --save student.pt    # or drift-demo; settings in configs/toy.yaml
pytest -q
python scripts/drift_curve.py                 # Proposition 1 bound against a simulated quantised scalar recurrence
```

## Toy run (real output)

The toy world is a moving band-limited texture. Its high-resolution frames stand in for teacher outputs, so the whole objective runs offline: LSQ, the state bank, the power-iteration penalty and the measurements. The demo trains a small student twice, with and without the contraction penalty, and evaluates both in W8A8 on a held-out 120-frame clip. Output of `python -m drift_vsr.demo` on one CPU thread (Python 3.12):

```
W8A8 student, 29,666 parameters; 150 FP + 150 QAT steps; held-out clip of 120 frames, x4
                                PSNR bilinear  drift sigma mean sigma max err first err last
DRIFT (contraction penalty)    25.77    23.94   0.16       0.92      1.01     0.587    0.579
same, without the penalty      26.53    23.94  -0.37       3.18      3.47     0.104    0.119
target gamma = 0.9; drift: PSNR of the first 10 frames minus the last 10 (dB); err: ||h_quant - h_float|| / ||h_float|| at the first and last frame
done in 66.7s
```

The penalty holds the state-Jacobian norm estimate near γ, where the unpenalised recurrence is expansive. In this toy that does not reduce the quantised state's error: the error is set at the first frame, changes little over the clip in either run, and is larger with the penalty. The toy therefore checks that the method runs end to end; it does not reproduce the report's drift result. `paper/toy_run.md` records the environment and what each line shows.

## Upscaling your own clip

`examples/upscale_clip.py` runs a saved student in W8A8 on a clip of frames (a NumPy array, `(T, H, W, 3)`), and reports the state-Jacobian norm along the clip, the quantised state's error relative to the float state, and, given a reference clip, PSNR against bilinear and drift. The bundled clip comes from the toy world; `student.pt` is the one the demo saved.

```bash
python examples/upscale_clip.py examples/data/clip_lr.npy --checkpoint student.pt --reference examples/data/clip_hr.npy
```

```
40 frames, 8x8 -> 32x32, W8A8 student (29,666 parameters)
state-Jacobian norm estimate: mean 0.94, max 0.99 (below 1 bounds the drift)
quantised state error relative to float: 0.589 at frame 1, 0.553 at frame 40
PSNR 26.10 dB against 24.05 dB for bilinear; drift@40 -0.35 dB (first 10 frames minus last 10)
```

The pieces from Python:

```python
from drift_vsr import DriftStudent, jacobian_spectral_norm, contraction_penalty, set_quant

model = DriftStudent(width=48, blocks_f=8, blocks_g=4)    # 1.9M parameters, the report's configuration
out, h, ctx = model(clip)                                 # clip: (B, T, 3, H, W) -> out: (B, T, 3, 4H, 4W)
h_prev, x, flow = ctx[-1]
set_quant(model, False)                                   # Phi_t is the real-valued map; rounding is eta_t
sigma, v = jacobian_spectral_norm(lambda s: model.transition(s, x, flow), h_prev, iters=2)
loss_c = contraction_penalty(sigma, gamma=0.9)
```

## Results (from the report)

These numbers come from the report's seeded simulation and are illustrative. Snapdragon 8 Gen 3 Hexagon NPU, 270p→1080p, mean over 3 seeds:

| Method | Params | VideoLQ DOVER ↑ | REDS30 LPIPS ↓ | drift@1000 ↓ | NPU ms/frame |
| --- | --- | --- | --- | --- | --- |
| RealBasicVSR (INT8 PTQ) | 6.3M | 0.672 | 0.262 | – | 118.0 |
| DOVE (teacher, A100 only) | 5.6B | 0.748 | 0.244 | – | does not fit |
| Student, W8A8 PTQ | 1.9M | 0.684 | 0.284 | 0.137 | 21.4 |
| Student, W8A8 QAT | 1.9M | 0.718 | 0.262 | 0.064 | 21.4 |
| **DRIFT W8A8** | 1.9M | **0.729** | **0.254** | **0.005** | 21.4 |
| DRIFT W4A8 | 1.9M | 0.707 | 0.271 | 0.014 | 14.9 |

`bash scripts/reproduce_paper.sh` regenerates the tables and figures that the report's seeded simulation produces. On every push, CI reruns `paper/make_data.py` and checks the tables and results it writes against `paper/results.sha256` (`python paper/check_results.py`).

Full training needs teacher outputs, made offline with the public DOVE checkpoint on REDS and YouHQ, and the LPIPS and discriminator terms. `configs/drift_x4.yaml` lists the hyperparameters. The toy path replaces LPIPS with a gradient-domain stand-in.

## Layout

| Path | What it holds |
| --- | --- |
| `src/drift_vsr/model.py` | Student, flow head, bilinear warp, gated contractive transition |
| `src/drift_vsr/quant.py` | LSQ fake quantiser and quantised convolution |
| `src/drift_vsr/contraction.py` | Power-iteration spectral-norm estimator, penalty, Proposition 1 bound |
| `src/drift_vsr/statebank.py` | State bank for long-rollout QAT |
| `src/drift_vsr/losses.py` | Distillation loss, consistency matching, residual suppression baseline |
| `src/drift_vsr/metrics.py` | drift@T, state error (absolute and relative to the state), state-Jacobian norm along a stream, PSNR |
| `src/drift_vsr/train.py`, `demo.py` | Algorithm 1 training loop and CLI; the toy run with and without the penalty |
| `configs/` | `toy.yaml` (CPU) and `drift_x4.yaml` (report) |
| `examples/` | `upscale_clip.py`, and a toy clip in `data/` |
| `scripts/` | `run_toy.py`, `reproduce_paper.sh`, `drift_curve.py` |
| `paper/` | `make_data.py`, `figures.py`, `check_results.py`, `results.sha256`, `toy_run.md` |

## Citation

```bibtex
@techreport{agarwal2025drift,
  title       = {{DRIFT}: Drift-Bounded Distillation of One-Step Video Diffusion into Integer Recurrent
                 Students for Real-Time On-Device Super-Resolution},
  author      = {Agarwal, Manav},
  institution = {hanaML},
  year        = {2025}
}
```

## License

Apache-2.0. See `LICENSE`.
