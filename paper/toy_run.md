# Toy run (real output)

A real run of the reference implementation on its CPU toy setting. Unlike the report's tables, which come from a
seeded simulation and are illustrative, every number below was printed by the code. The toy is small by design: it
checks that the method works end to end on a problem with a known answer; it does not reproduce the report's
experiments.

- Command: `python -m drift_vsr.demo --save student.pt` (settings in `configs/toy.yaml`)
- Code: commit `a4293f7`
- Environment: Python 3.12.13, numpy 2.5.4, torch 2.14.1+cpu, PyYAML 6.0.3; Linux x86_64, Intel(R) Core(TM) i9-14900KF
- Wall time: 71 s on a shared machine under heavy load (1-minute load average 95 on 32 hardware threads); CPU time 72 s

## The toy setting

The toy world (`src/drift_vsr/toy.py`) is a band-limited random texture translated with a slowly varying velocity;
low-resolution inputs are downsampled four times by average pooling and noised, and the high-resolution frames stand in
for the teacher's outputs, so the whole objective runs offline. The student has 16 channels and 29,666 parameters
and runs in W8A8 (LSQ fake quantisation of weights, activations and the recurrent state). Each run trains on one CPU thread for 150
full-precision steps and then 150 steps of long-rollout QAT with the state bank, on four 60-frame videos; the two
runs differ only in the penalty on the state-Jacobian norm (gamma = 0.9, weight 0.1, or no penalty). Both are
evaluated in W8A8 on a held-out 120-frame clip: PSNR against bilinear upsampling; drift, the PSNR of the first ten
frames minus the last ten; a power-iteration estimate of the state-Jacobian norm of the real-valued transition at
every fourth frame; and the quantised state's error relative to the float state at the first and last frames.

## Output

```
W8A8 student, 29,666 parameters; 150 FP + 150 QAT steps; held-out clip of 120 frames, x4
                                PSNR bilinear  drift sigma mean sigma max err first err last
DRIFT (contraction penalty)    25.77    23.94   0.16       0.92      1.01     0.587    0.579
same, without the penalty      26.53    23.94  -0.37       3.18      3.47     0.104    0.119
target gamma = 0.9; drift: PSNR of the first 10 frames minus the last 10 (dB); err: ||h_quant - h_float|| / ||h_float|| at the first and last frame
done in 66.7s
```

## What it shows

- Both students beat bilinear upsampling (25.77 and 26.53 dB against 23.94); the penalty costs 0.76 dB.
- The penalty does what it targets: the Jacobian norm estimate stays near gamma (mean 0.92, at most 1.01), while
  without it the recurrence is expansive (mean 3.18, at most 3.47).
- In this toy that does not reduce the error it is meant to bound. The quantised state's relative error is set at
  the first frame and changes little over the clip in either run (0.587 to 0.579 with the penalty, 0.104 to 0.119
  without), and it is larger with the penalty. Neither run loses quality over the clip (drift 0.16 and -0.37 dB).
- So the toy does not reproduce the report's drift result over long streams. It checks that the training loop, the
  state bank, the penalty and the measurements run end to end; the report's numbers come from its seeded
  simulation.
- The toy trains on one thread because its results depend on the thread count: with four threads the same
  configuration gives slightly different numbers (25.72 dB, a Jacobian norm estimate of mean 0.84) and the same
  qualitative picture.
