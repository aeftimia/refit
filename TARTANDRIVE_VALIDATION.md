# TartanDrive 2.0 validation

This experiment evaluates the repository's dense optical-flow speed shape on
the synchronized TartanDrive 2.0 sequence
`2023-11-14-14-24-21_gupta`.

The public sequence provides:

- 10 Hz forward RGB images from a Carnegie Robotics MultiSense S21;
- 50 Hz fused NovAtel GNSS/INS odometry;
- four 50 Hz Racepak wheel encoders, ordered front-left, front-right,
  rear-left, rear-right.

The odometry speed is the dry comparison. Optical flow uses the production
preprocessing and Farneback median-ROI magnitude, subtracts a stationary
baseline found without consulting wheel data, and is normalized to the
odometry mean. No temporal smoothing is applied. Wheel measurements are held
out for evaluation.

The public RGB stream is 10 Hz, so each optical pair spans 0.1 seconds rather
than the roughly 1/30--1/60 second pair used by typical action-camera footage.
Mean normalization removes the unit-scale difference, but larger inter-frame
displacements may still make Farneback less accurate; the SFU 30 Hz experiment
is the closer frame-rate validation once that dataset is reachable.

## Run

Install the normal project requirements, then run:

```bash
python tartandrive_validation.py
```

The script downloads only the selected RGB, odometry, wheel, and metadata
objects from the official TartanDrive object store (about 684 MiB), extracts
the images, runs optical flow in parallel, and writes:

- `validation-output/tartandrive2/report.md`
- `validation-output/tartandrive2/metrics.json`
- `validation-output/tartandrive2/comparison.csv`

Downloaded data defaults to `validation-data/tartandrive2`. Both data and
output locations are configurable.

## Interpretation

The primary wheel trace is the sample-wise median of all four encoders. This
is robust to a single spinning, dropping, or faulty wheel. The report includes
each wheel's disagreement from that median and a rear-wheel-pair sensitivity
comparison, so conclusions are based on measured disagreement rather than an
assumption that ATV slip is intrinsically worse than mountain-bike slip.

Absolute wheel speed is also reported using the Yamaha Viking's nominal
25-inch tire diameter. The main method comparison mean-normalizes the wheel
trace because loaded tire radius and the Racepak calibration are not published
for this particular modified ATV. That comparison therefore tests temporal
speed-profile accuracy, not absolute wheel circumference calibration.
