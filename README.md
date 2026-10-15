# DRIFT: drift-bounded distillation for on-device video super-resolution

[![ci](https://github.com/hanaml-lab/ondevice-video-sr/actions/workflows/ci.yml/badge.svg)](https://github.com/hanaml-lab/ondevice-video-sr/actions/workflows/ci.yml)

Reference implementation of *DRIFT: Drift-Bounded Distillation of Video Diffusion into Integer Recurrent Students for Real-Time On-Device Super-Resolution* (Agarwal, hanaML preprint, 2024).

When a recurrent video super-resolution network runs in INT8, rounding error enters the hidden state every frame
and builds up over the stream. DRIFT distils a video diffusion teacher (Upscale-A-Video) into a 1.9M-parameter
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

The toy world is a moving band-limited texture. Its high-resolution frames stand in for teacher outputs, so the whole objective runs offline: LSQ, the state bank, the power-iteration penalty and the measurements. The demo trains a small student twice on one CPU thread, with and without the contraction penalty, and runs each over a held-out stream of 2,000 frames. Beside the W8A8 student it runs the real-valued recurrence and a copy that rounds only the recurrent state, and measures what Proposition 1 is about: eps, the largest error one step injects into the state; gamma*, the largest factor by which one step carries the accumulated error forward; and the accumulated error itself, early and late in the stream. Output of `python -m drift_vsr.demo` (Python 3.12):

```
W8A8 student, 29,666 parameters; 150 FP + 150 QAT steps on 60-frame videos; held-out stream of 2,000 frames, x4
                                PSNR  bilinear  drift  sigma mean  sigma max
DRIFT (contraction penalty)    25.78     23.98  -0.57        0.93       0.97
same, without the penalty      26.61     23.98  -0.68        3.19       3.43

Accumulated state error ||h_round - h_float||, in units of eps, mean over frames:
                                  eps  gamma*  1/(1-gamma*)    1-10  91-100  991-1000  1991-2000    max
DRIFT, state rounded only      2.2879    0.15           1.2    0.98    0.79      0.94       0.95   1.03
DRIFT, W8A8                    2.3050    0.16           1.2    0.99    0.80      0.94       0.95   1.03
no penalty, state rounded only 0.0428    0.50           2.0    1.01    1.02      1.02       1.03   1.13
no penalty, W8A8               0.1446    0.42           1.7    0.95    0.84      0.90       0.94   1.08
eps: the largest error one step injects; gamma*: the largest factor by which one step carries the error forward. Proposition 1 bounds the error by eps/(1-gamma*) when gamma* < 1 (target gamma = 0.9).
drift: PSNR of the first 10 frames minus the last 10 (dB); sigma: state-Jacobian norm estimate at every 100th frame
done in 52.9s
```

With the penalty the error stays bounded, as Proposition 1 says it must: no step carries more than 0.16 of it forward, so it never exceeds about one step's rounding (1.03 eps, under the bound of 1.2 eps). Without the penalty it does not accumulate either. The unpenalised student's state-Jacobian norm estimate averages 3.19, yet no step carries more than half of the actual error forward (gamma* 0.50), and after 2,000 frames the error is still about one step's rounding. So the toy does not reproduce the report's drift result: its unpenalised student already contracts the rounding error where it matters, and there is no accumulation for the penalty to remove. Seeds 1 and 2 give the same picture. The toy checks that the bound holds where its premise does, that the measurement is right and that the method runs end to end. `paper/toy_run.md` records the environment, what was fixed before the run, the other seeds and what each line shows.

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
| Upscale-A-Video (teacher, A100 only) | 1.1B | 0.714 | 0.271 | – | does not fit |
| Student, W8A8 PTQ | 1.9M | 0.650 | 0.284 | 0.137 | 21.4 |
| Student, W8A8 QAT | 1.9M | 0.684 | 0.262 | 0.064 | 21.4 |
| **DRIFT W8A8** | 1.9M | **0.695** | **0.254** | **0.005** | 21.4 |
| DRIFT W4A8 | 1.9M | 0.673 | 0.271 | 0.014 | 14.9 |

`bash scripts/reproduce_paper.sh` regenerates the tables and figures that the report's seeded simulation produces. On every push, CI reruns `paper/make_data.py` and checks the tables and results it writes against `paper/results.sha256` (`python paper/check_results.py`).

Full training needs teacher outputs, made offline with the public Upscale-A-Video release (30 steps) on REDS and YouHQ, and the LPIPS and discriminator terms. `configs/drift_x4.yaml` lists the hyperparameters. The toy path replaces LPIPS with a gradient-domain stand-in.

## Layout

| Path | What it holds |
| --- | --- |
| `src/drift_vsr/model.py` | Student, flow head, bilinear warp, gated contractive transition |
| `src/drift_vsr/quant.py` | LSQ fake quantiser and quantised convolution |
| `src/drift_vsr/contraction.py` | Power-iteration spectral-norm estimator, penalty, Proposition 1 bound |
| `src/drift_vsr/statebank.py` | State bank for long-rollout QAT |
| `src/drift_vsr/losses.py` | Distillation loss, consistency matching, residual suppression baseline |
| `src/drift_vsr/metrics.py` | drift@T, state error (absolute and relative to the state), Proposition 1's quantities along a stream (`rounding_decomposition`), state-Jacobian norm along a stream, PSNR |
| `src/drift_vsr/train.py`, `demo.py` | Algorithm 1 training loop and CLI; the toy run with and without the penalty |
| `configs/` | `toy.yaml` (CPU) and `drift_x4.yaml` (report) |
| `examples/` | `upscale_clip.py`, and a toy clip in `data/` |
| `scripts/` | `run_toy.py`, `reproduce_paper.sh`, `drift_curve.py` |
| `paper/` | `make_data.py`, `figures.py`, `check_results.py`, `results.sha256`, `toy_run.md` |

## Citation

```bibtex
@techreport{agarwal2024drift,
  title       = {{DRIFT}: Drift-Bounded Distillation of Video Diffusion into Integer Recurrent
                 Students for Real-Time On-Device Super-Resolution},
  author      = {Agarwal, Manav},
  institution = {hanaML},
  year        = {2024}
}
```

## License

Apache-2.0. See `LICENSE`.
