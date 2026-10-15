"""Trajectory-level evaluation: drift@T, the state error under quantisation and its decomposition (Proposition 1),
and the state-Jacobian norm along a stream."""
from __future__ import annotations

from typing import Callable, Sequence

import torch

from .quant import set_quant


def drift_at(scores: Sequence[float], T: int, k: int = 10) -> float:
    """drift@T = mean score of frames 1..k minus mean score of frames T-k+1..T (higher = worse)."""
    if T > len(scores) or T < k:
        raise ValueError(f"need k <= T <= len(scores); got T={T}, k={k}, len={len(scores)}")
    head = sum(scores[:k]) / k
    tail = sum(scores[T - k:T]) / k
    return float(head - tail)


@torch.no_grad()
def state_error_trajectory(model, stream: torch.Tensor) -> list[float]:
    """||h~_t - h_t|| per frame: run the same stream in float and in fake-quant mode (Proposition 1)."""
    out = []
    set_quant(model, False)
    hf = model.init_state(stream[:, 0])
    hq = hf.clone()
    xp = stream[:, 0]
    for t in range(stream.shape[1]):
        x = stream[:, t]
        set_quant(model, False)
        hf, _, _ = model.step(hf, xp, x)
        set_quant(model, True)
        hq, _, _ = model.step(hq, xp, x)
        out.append(float((hq - hf).flatten(1).norm(dim=1).mean()))
        xp = x
    set_quant(model, True)
    return out


def relative_state_error(model, stream: torch.Tensor) -> list[float]:
    """||h~_t - h_t|| / ||h_t|| per frame: the quantised state's error relative to the size of the float state."""
    out = []
    set_quant(model, False)
    hf = model.init_state(stream[:, 0])
    hq = hf.clone()
    xp = stream[:, 0]
    with torch.no_grad():
        for t in range(stream.shape[1]):
            x = stream[:, t]
            set_quant(model, False)
            hf, _, _ = model.step(hf, xp, x)
            set_quant(model, True)
            hq, _, _ = model.step(hq, xp, x)
            err = (hq - hf).flatten(1).norm(dim=1) / hf.flatten(1).norm(dim=1).clamp_min(1e-12)
            out.append(float(err.mean()))
            xp = x
    set_quant(model, True)
    return out


def quant_mode(model, mode: str) -> None:
    """Which rounding runs: "float" (none), "state" (only the recurrent state, 8-bit) or "w8a8" (every quantiser,
    as deployed)."""
    if mode not in ("float", "state", "w8a8"):
        raise ValueError(f"mode must be float, state or w8a8; got {mode!r}")
    set_quant(model, mode == "w8a8")
    if mode == "state":
        model.state_q.enabled = True


@torch.no_grad()
def rounding_decomposition(model, stream: torch.Tensor, modes: Sequence[str] = ("state", "w8a8"),
                           ref: torch.Tensor | None = None) -> dict[str, dict[str, list]]:
    """Proposition 1 measured along a stream.

    Runs the real-valued recurrence h_t = Phi_t(h_{t-1}) and, for each mode, the rounded one h~_t = Phi_t(h~_{t-1})
    + eta_t side by side from the same initial state ("state" rounds only the recurrent state, to 8 bits; "w8a8" runs
    every quantiser, as deployed). Returns, per mode and per frame:

    - ``err``: ||h~_t - h_t||, the accumulated error;
    - ``inj``: ||eta_t||, the rounding step t injects: the rounded step minus the real-valued map Phi_t applied to the
      same state h~_{t-1};
    - ``gain``: ||Phi_t(h~_{t-1}) - Phi_t(h_{t-1})|| / ||h~_{t-1} - h_{t-1}||, the factor by which step t carries the
      existing error forward (None at the first frame, where there is no error yet);
    - ``norm``: ||h_t||, the size of the real-valued state;
    - with ``ref`` (reference frames at the output resolution) and mode "w8a8": ``psnr`` of the deployed outputs.

    By the triangle inequality err_t <= gain_t * err_{t-1} + inj_t, so if every inj_t <= eps and every gain_t <= gamma
    < 1, then err_t <= eps / (1 - gamma) at every frame.
    """
    for m in modes:
        if m not in ("state", "w8a8"):
            raise ValueError(f"modes are state and w8a8; got {m!r}")

    def size(v: torch.Tensor) -> float:
        return float(v.flatten(1).norm(dim=1).mean())

    hf = model.init_state(stream[:, 0])
    hq = {m: hf.clone() for m in modes}
    out = {m: {"err": [], "inj": [], "gain": [], "norm": [], "psnr": []} for m in modes}
    xp = stream[:, 0]
    b = hf.shape[0]
    for t in range(stream.shape[1]):
        x = stream[:, t]
        quant_mode(model, "float")
        flow = model.motion(xp, x)
        # Phi_t at the real-valued state and at each rounded state, in one batch
        k = 1 + len(modes)
        phi = model.transition(torch.cat([hf] + [hq[m] for m in modes]), x.repeat(k, 1, 1, 1), flow.repeat(k, 1, 1, 1))
        hf_next = phi[:b]
        for i, m in enumerate(modes):
            phi_q = phi[b * (i + 1):b * (i + 2)]
            quant_mode(model, m)
            if m == "state":  # with only the state quantiser on, the step is Phi_t followed by the state's rounding
                hq_next = model.state_q(phi_q)
            else:
                hq_next, y, _ = model.step(hq[m], xp, x)
                if ref is not None:
                    out[m]["psnr"].append(psnr(y, ref[:, t]))
            prev = size(hq[m] - hf)
            out[m]["gain"].append(size(phi_q - hf_next) / prev if t > 0 and prev > 0 else None)
            out[m]["inj"].append(size(hq_next - phi_q))
            out[m]["err"].append(size(hq_next - hf_next))
            out[m]["norm"].append(size(hf_next))
            hq[m] = hq_next
        hf, xp = hf_next, x
    set_quant(model, True)
    return out


def state_jacobian_norms(model, stream: torch.Tensor, iters: int = 8, stride: int = 1) -> list[float]:
    """Power-iteration estimate of ||d h_t / d h_{t-1}||_2 of the real-valued transition along the float trajectory
    (Phi_t in Proposition 1), at every ``stride``-th frame."""
    from .contraction import jacobian_spectral_norm

    out = []
    set_quant(model, False)
    with torch.no_grad():
        h = model.init_state(stream[:, 0])
    xp = stream[:, 0]
    for t in range(stream.shape[1]):
        x = stream[:, t]
        if t % stride == 0:
            sigma, _ = jacobian_spectral_norm(lambda s, xp=xp, x=x: model.step(s, xp, x)[0], h, iters=iters)
            out.append(float(sigma.detach().mean()))
        with torch.no_grad():
            h, _, _ = model.step(h, xp, x)
        xp = x
    set_quant(model, True)
    return out


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.clamp(0, 1) - b.clamp(0, 1)).pow(2).mean()
    return float(10 * torch.log10(1.0 / mse.clamp_min(1e-12)))


def per_frame(fn: Callable[[torch.Tensor, torch.Tensor], float], pred: torch.Tensor, ref: torch.Tensor) -> list[float]:
    return [fn(pred[:, t], ref[:, t]) for t in range(pred.shape[1])]
