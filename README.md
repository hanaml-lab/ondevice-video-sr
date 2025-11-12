# DRIFT: drift-bounded distillation for on-device video super-resolution

Reference implementation for *DRIFT: Drift-Bounded Distillation of One-Step Video Diffusion into Integer
Recurrent Students for Real-Time On-Device Super-Resolution* (Manav Agarwal, hanaML, 2025).

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
pip install -e ".[dev,paper]"      # torch, numpy, pyyaml; pytest and matplotlib optional
```

## Quickstart (CPU, under a minute)

```bash
bash scripts/train_toy.sh          # FP warm-up, then long-rollout QAT with the contraction penalty
python -m drift_vsr.train --config configs/toy.yaml --no-contraction   # the same run without the penalty
python scripts/drift_curve.py      # Proposition 1 bound vs. a simulated quantised recurrence
pytest -q                          # 9 tests
```

The toy world is a moving band-limited texture. Its high-resolution frames stand in for teacher outputs, so the
whole objective runs offline: LSQ, the state bank, the power-iteration penalty and the drift metric. On one CPU
with 4 threads the default run takes about 21 s. It reaches 26.1 dB PSNR, against 23.9 dB for bilinear
upsampling, and ends with a state-Jacobian norm estimate of 0.79, below the target γ = 0.9. The toy only checks
that the method runs end to end. It does not reproduce the paper.

```python
from drift_vsr import DriftStudent, jacobian_spectral_norm, contraction_penalty, set_quant

model = DriftStudent(width=48, blocks_f=8, blocks_g=4)    # 1.9M parameters, the paper configuration
out, h, ctx = model(clip)                                 # clip: (B, T, 3, H, W) -> out: (B, T, 3, 4H, 4W)
h_prev, x, flow = ctx[-1]
set_quant(model, False)                                   # Phi_t is the real-valued map; rounding is eta_t
sigma, v = jacobian_spectral_norm(lambda s: model.transition(s, x, flow), h_prev, iters=2)
loss_c = contraction_penalty(sigma, gamma=0.9)
```

## Layout

| Path | What it holds |
| --- | --- |
| `src/drift_vsr/model.py` | Student, flow head, bilinear warp, gated contractive transition |
| `src/drift_vsr/quant.py` | LSQ fake quantiser and quantised convolution |
| `src/drift_vsr/contraction.py` | Power-iteration spectral-norm estimator, penalty, Proposition 1 bound |
| `src/drift_vsr/statebank.py` | State bank for long-rollout QAT |
| `src/drift_vsr/losses.py` | Distillation loss, consistency matching, residual suppression baseline |
| `src/drift_vsr/metrics.py` | drift@T, float-vs-quantised state-error trajectory, PSNR |
| `src/drift_vsr/train.py` | Algorithm 1 training loop and CLI |
| `configs/` | `toy.yaml` (CPU) and `drift_x4.yaml` (paper) |
| `paper/` | `make_data.py` and `figures.py`, which produce every table and figure |
| `scripts/` | Toy training, figure and table reproduction, bound simulation |

## Results (paper)

The numbers come from the paper, on a Snapdragon 8 Gen 3 Hexagon NPU, 270p→1080p, mean over 3 seeds. Run
`python scripts/reproduce_paper.py` to regenerate all of its tables and figures.

| Method | Params | VideoLQ DOVER ↑ | REDS30 LPIPS ↓ | drift@1000 ↓ | NPU ms/frame |
| --- | --- | --- | --- | --- | --- |
| RealBasicVSR (INT8 PTQ) | 6.3M | 0.672 | 0.262 | – | 118.0 |
| DOVE (teacher, A100 only) | 5.6B | 0.748 | 0.244 | – | does not fit |
| Student, W8A8 PTQ | 1.9M | 0.684 | 0.284 | 0.137 | 21.4 |
| Student, W8A8 QAT | 1.9M | 0.718 | 0.262 | 0.064 | 21.4 |
| **DRIFT W8A8** | 1.9M | **0.729** | **0.254** | **0.005** | 21.4 |
| DRIFT W4A8 | 1.9M | 0.707 | 0.271 | 0.014 | 14.9 |

Full training needs teacher outputs, made offline with the public DOVE checkpoint on REDS and YouHQ, and the
LPIPS and discriminator terms. `configs/drift_x4.yaml` lists the hyperparameters. The toy path replaces LPIPS
with a gradient-domain stand-in.

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

## Licence

Apache-2.0. See `LICENSE`.
