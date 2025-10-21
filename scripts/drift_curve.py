"""Plot the Proposition 1 bound against a simulated quantised scalar recurrence for several gamma."""
import argparse

import numpy as np

from drift_vsr import drift_bound

p = argparse.ArgumentParser()
p.add_argument("--eps", type=float, default=0.01)
p.add_argument("--frames", type=int, default=1000)
a = p.parse_args()
rng = np.random.default_rng(0)
for g in (0.8, 0.9, 0.95, 0.99, 1.0):
    d, worst = 0.0, 0.0
    for t in range(1, a.frames + 1):
        d = g * d + rng.uniform(-a.eps, a.eps)
        worst = max(worst, abs(d))
    print(f"gamma={g:<5} max|delta|={worst:.4f}  bound={drift_bound(a.eps, g, t=a.frames):.4f}")
