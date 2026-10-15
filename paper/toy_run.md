# Toy run (real output)

A real run of the reference implementation on its CPU toy setting. Unlike the report's tables, which come from a
seeded simulation and are illustrative, every number below was printed by the code. The toy is small by design: it
checks the method end to end on a problem small enough for a CPU; it does not reproduce the report's experiments.

- Command: `python -m drift_vsr.demo --save student.pt` (settings in `configs/toy.yaml`)
- Code: commit `71e6bb1`
- Environment: Python 3.12.13, numpy 2.3.5, torch 2.14.1+cpu, PyYAML 6.0.3; Linux x86_64, Intel(R) Core(TM) i9-14900KF
- Wall time: 56 s on one thread on a shared machine (1-minute load average 23 on 32 hardware threads); CPU time 57 s

## What the toy tests, fixed before it was run

The report's claim is that rounding error in a quantised recurrent state accumulates without contraction and stays
bounded by eps/(1 - gamma) with it, over streams of thousands of frames. The toy's measurement and what would count
as reproducing that claim were fixed before the first run, so that the toy could not be tuned until it agreed:

- Training is unchanged from the earlier toy (`configs/toy.yaml`: 150 full-precision and 150 long-rollout QAT steps
  on four 60-frame videos; the two runs differ only in the contraction penalty, gamma = 0.9, weight 0.1).
- Each student runs over a held-out stream of 2,000 frames, far beyond the 60 frames it was trained on, three ways in
  lockstep: the real-valued recurrence h_t = Phi_t(h_{t-1}); a copy that rounds only the recurrent state to 8 bits;
  and the W8A8 student as deployed. For each rounded copy the demo measures the error h~_t - h_t; the rounding step t
  injects, eta_t = h~_t - Phi_t(h~_{t-1}), with eps the largest; and the factor by which step t carries the error
  forward, ||Phi_t(h~_{t-1}) - Phi_t(h_{t-1})|| / ||h~_{t-1} - h_{t-1}||, with gamma* the largest. By the triangle
  inequality the error is then at most eps/(1 - gamma*) at every frame whenever gamma* < 1 (Proposition 1).
- The claim would count as reproduced if, on seeds 0, 1 and 2: with the penalty, gamma* < 1 for both roundings (so
  the bound holds); and without it, the error of the state-only rounding over frames 1991-2000 is at least twice its
  level over frames 91-100 (the error accumulates).

The stream was first planned at 3,000 frames and the Jacobian estimate at every 20th frame; both were reduced, before
the first run, to keep the demo near a minute on one thread.

## Output

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

## What it shows

- Both students beat bilinear upsampling (25.78 and 26.61 dB against 23.98 dB), and neither loses quality over the
  stream (drift -0.57 and -0.68 dB: the last ten frames score slightly higher than the first ten).
- With the penalty the error is bounded, as Proposition 1 says it must be: no step carries more than 0.16 of the
  accumulated error forward (gamma* 0.15 with the state rounded alone, 0.16 in W8A8), so the error never exceeds
  about one step's rounding (at most 1.03 eps, under the bound of 1.2 eps).
- Without the penalty the error does not accumulate either. The unpenalised student's state-Jacobian norm estimate
  averages 3.19, yet no step carries more than half of the actual error forward (gamma* 0.50 with the state rounded
  alone, 0.42 in W8A8), and after 2,000 frames the error is still about one step's rounding (1.03 eps over frames
  1991-2000 against 1.02 over frames 91-100). A Jacobian norm above 1 means the bound does not apply; along these
  streams it does not make the error grow.
- So the toy does not reproduce the report's claim: its unpenalised student already contracts the rounding error
  where it matters, and there is no accumulation for the penalty to remove. The penalty also makes each step's
  rounding larger (eps 2.29 against 0.043 with the state rounded alone): the penalised student's 8-bit state
  quantiser has a step of 0.00087, and about a fifth of its state's elements fall outside that range and are clipped.
- What the toy does check: that the bound holds wherever its premise does, that the measurement of Proposition 1's
  quantities is right (a test checks the one-step recursion), and that training, the state bank, the penalty and the
  measurements run end to end. The report's drift numbers come from its seeded simulation.

## Seeds 1 and 2

The same command with `seed: 1` and `seed: 2` in a copy of `configs/toy.yaml` gives the same picture: with the
penalty gamma* is at most 0.35; without it gamma* is at most 0.45 and the error over frames 1991-2000 equals its level
over frames 91-100 to within 0.01 eps.

```
W8A8 student, 29,666 parameters; 150 FP + 150 QAT steps on 60-frame videos; held-out stream of 2,000 frames, x4
                                PSNR  bilinear  drift  sigma mean  sigma max
DRIFT (contraction penalty)    26.39     24.36   0.84        0.81       0.89
same, without the penalty      27.19     24.36   0.79        2.50       2.70

Accumulated state error ||h_round - h_float||, in units of eps, mean over frames:
                                  eps  gamma*  1/(1-gamma*)    1-10  91-100  991-1000  1991-2000    max
DRIFT, state rounded only      0.0741    0.17           1.2    0.97    0.96      0.97       0.97   1.01
DRIFT, W8A8                    3.4914    0.14           1.2    0.92    0.92      0.87       0.98   1.02
no penalty, state rounded only 0.0697    0.45           1.8    1.00    1.02      1.01       1.02   1.10
no penalty, W8A8               0.1382    0.39           1.7    0.94    0.94      0.95       0.99   1.11
eps: the largest error one step injects; gamma*: the largest factor by which one step carries the error forward. Proposition 1 bounds the error by eps/(1-gamma*) when gamma* < 1 (target gamma = 0.9).
drift: PSNR of the first 10 frames minus the last 10 (dB); sigma: state-Jacobian norm estimate at every 100th frame
done in 59.5s
```

```
W8A8 student, 29,666 parameters; 150 FP + 150 QAT steps on 60-frame videos; held-out stream of 2,000 frames, x4
                                PSNR  bilinear  drift  sigma mean  sigma max
DRIFT (contraction penalty)    27.27     24.36  -0.37        0.94       1.09
same, without the penalty      27.31     24.36  -0.23        1.22       1.45

Accumulated state error ||h_round - h_float||, in units of eps, mean over frames:
                                  eps  gamma*  1/(1-gamma*)    1-10  91-100  991-1000  1991-2000    max
DRIFT, state rounded only      0.0436    0.33           1.5    0.99    1.00      1.00       1.00   1.05
DRIFT, W8A8                    0.1701    0.35           1.5    1.02    1.06      1.08       0.99   1.13
no penalty, state rounded only 0.0202    0.38           1.6    1.01    1.02      1.01       1.01   1.07
no penalty, W8A8               0.1413    0.33           1.5    0.90    0.90      0.99       0.93   1.06
eps: the largest error one step injects; gamma*: the largest factor by which one step carries the error forward. Proposition 1 bounds the error by eps/(1-gamma*) when gamma* < 1 (target gamma = 0.9).
drift: PSNR of the first 10 frames minus the last 10 (dB); sigma: state-Jacobian norm estimate at every 100th frame
done in 62.4s
```

The toy trains on one thread because its results depend on the thread count. On one thread a run repeats exactly: a
second run of the command above printed the same output (in 118 s, on a more heavily loaded machine).
