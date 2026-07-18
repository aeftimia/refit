# SFU Mountain validation

This experiment compares the project's two speed models against synchronized
wheel-encoder velocity on the public, CC BY 4.0
[SFU Mountain Dataset](https://autonomy.cs.sfu.ca/sfu-mountain-dataset/).

The `dry-b` daylight traversal was chosen because it uses the lower-grade Garmin
18x receiver (5 Hz, quoted 15 m accuracy), a forward global-shutter camera, and
10 Hz wheel odometry. The dataset authors shifted the camera and wheel timestamps
onto a common clock. They explicitly note that GPS could not be latency-aligned,
so the adapter estimates a static GPS/camera offset from optical/GPS rank
correlation without using wheel truth.

## Reproduce

Install the project requirements plus `aria2c` (`brew install aria2` on macOS),
then run:

```bash
python3 sfu_mountain_validation.py --download
```

This downloads only three files from the 522 GB Academic Torrents release: the
3 Hz forward-left JPEG archive (about 405 MiB), GPS latitude/longitude, and
wheel odometry. Dataset payloads belong under `validation-data/` and are not
committed. Results are written to `validation-output/sfu-mountain/`.

The official server also provides the full 30 Hz camera archive. For a faithful
comparison to high-frame-rate action-camera footage, use:

```bash
python3 sfu_mountain_validation.py --download --camera-rate 30
```

That camera archive is about 4.0 GiB. The 3 Hz run is a cheaper pipeline and
domain smoke test; its larger consecutive-frame displacement is not equivalent
to the project's normal 30 fps frame-pair measurement.

## Method

1. Parse timestamp-named forward images and header-labelled GPS/wheel CSV files.
2. Select a two-minute interval with GPS speed variation, without consulting
   wheel truth.
3. Derive dry speed from consecutive GPS arcs.
4. Estimate dense Farneback median flow from consecutive camera frames using the
   production ROI and preprocessing, without temporal smoothing.
5. Align the unsynchronized GPS clock to camera motion by rank correlation.
6. Subtract the quietest stop-interval median only if coordinate-derived GPS
   contains a genuine, nonzero-duration zero-speed interval; otherwise use the
   production fallback of exactly zero.
7. Scale optical motion to the GPS-derived distance over the same interval.
8. Compare both estimates with wheel velocity at 10 Hz using MAE, RMSE, P95,
   maximum absolute error, bias, and correlations.

Wheel velocity is used only in step 8. It does not select the test interval,
clock offset, baseline, or optical scale.

## Availability caveat

As of 2026-07-18, the university HTTP/HTTPS file host timed out and the Academic
Torrents swarm exposed peers but no seed delivered data. The adapter and selective
downloader are ready, but no numeric result should be claimed until one of those
public mirrors becomes available. The torrent hash is
`e3d6b8d9e87cab68c7947e800e337e58fc8d8e59`.
