# FIT speed correction for Insta360 telemetry

This project aligns a video clock with an original Garmin FIT activity and
writes an Insta360-compatible FIT file. The writer patches fixed-width FIT
timestamp and speed fields in the original binary and recalculates its CRCs;
every non-target byte is retained, including Garmin messages, developer fields,
GPS metadata, heart rate, elevation, events, and device information. By default,
the command synchronizes timestamps while keeping Garmin speeds. Optical speed
replacement is slower and requires an explicit `--full` flag.

The shortest workflow needs only the video. The first run prompts securely for
Garmin Connect credentials and MFA when required; later runs reuse refreshable
tokens stored in `~/.garminconnect`. The matching original FIT is cached under
`~/.cache/refit/garmin`, then passed through the same lossless transformation.
The launcher automatically uses the repository's `venv/bin/python` when it
exists, so it works even when another Conda or system Python is active. Set
`REFIT_PYTHON` to override the interpreter.

```bash
bash insta360_video_speed_fit.sh VIDEO.mp4
```

The default output is `<video-basename>_speed.fit` in the directory where the
command is run. For example, running against `/Volumes/Camera/VID_001.mp4` from
`~/Downloads` writes `~/Downloads/VID_001_speed.fit`.

Local files and explicit output paths remain supported:

```bash
bash insta360_video_speed_fit.sh VIDEO.mp4 GARMIN.fit OUTPUT.fit
bash insta360_video_speed_fit.sh --output OUTPUT.fit VIDEO.mp4
bash insta360_video_speed_fit.sh --full VIDEO.mp4 GARMIN.fit OUTPUT.fit
```

Automatic selection searches Garmin activities around the video date and picks
the activity with the greatest timeline overlap. Use `--activity-id ID` to
override selection, or `--token-store DIR` to isolate the authorization cache.
The Garmin Connect downloader uses Garmin's mobile authentication flow through
the third-party `garminconnect` package; it is separate from Garmin's official
Activity API, which requires Developer Program approval for a registered cloud
integration.

## Optical speed pipeline

1. Uniformly sample adjacent video-frame pairs.
2. Downscale each frame to 640 pixels wide and convert it to grayscale.
3. Compute dense Farneback optical flow directly between the unblurred,
   full-frame grayscale source frames `N` and `N+1`.
   Sampling controls how often one of these adjacent-frame pairs is measured;
   it does not increase the time separating the two frames in a pair. For
   example, 4 Hz analysis of 60 fps video uses approximately `(0, 1)`,
   `(15, 16)`, `(30, 31)`, and so on.
4. Convert each per-pixel flow vector to magnitude.
5. Select the central spatial ROI and reduce it to its median magnitude.
6. Preserve the resulting scalar motion series without temporal smoothing.
   Spatial median reduction already rejects pixel outliers, while an assumed
   temporal cutoff could erase genuine acceleration and braking.
7. Align raw optical motion with Garmin's approximately 1 Hz `gps_metadata`
   speed stream using a static clock-shift search and Spearman rank correlation.
8. Identify contiguous, nonzero-duration intervals where the aligned FIT reports
   exactly zero speed.
9. Calculate the median optical magnitude within each observed stop. Use the
   lowest interval median as the additive optical baseline. This selects the
   quietest observed stop without relying on the lowest individual frame.
10. If no stationary interval exists, use a baseline of exactly zero. The code
    does not assume the camera is stationary for any fixed fraction of a ride.
11. Subtract the baseline and clamp negative results to zero.
12. Scale the corrected optical series so its time integral equals the Garmin
    coordinate distance over the video/FIT overlap.
13. Patch both record `enhanced_speed` and the denser Garmin GPS-metadata speed
    values while preserving every other original FIT field.
14. Encode the clock correction in whole-second FIT timestamps and apply any
    fractional remainder to synthetic-speed sampling, preserving subsecond
    video alignment without modifying the source video.

Gaussian blur and padded early cropping were tested on TartanDrive and removed.
The simplest full-frame, unblurred estimator produced the best wheel-referenced
MAE and correlation. Early cropping improved isolated flow throughput but
slightly worsened accuracy, changed the signal during large motion, and had
little effect on dry-run wall time because video seeking dominates.

## Experimental clock drift

Static clock correction remains the default. To fit a linear clock-rate error
as well as the starting offset, run:

```bash
bash insta360_video_speed_fit.sh --clock-drift VIDEO.mp4
```

The affine model maps video time to FIT time as
`FIT = video + start_offset + rate * video_elapsed`. The command searches rates
from an initial 5000 ppm bound, expands that bound whenever the optimum lands
on it, reports the fitted rate and start/end corrections, and
non-uniformly warps original FIT timestamps onto the MP4 clock while preserving
sensor values. Use `--clock-drift-ppm N` to set a different explicit search
bound without automatic expansion. Correlation values are optimization
diagnostics, not calibrated confidence measures. A solution that remains at the
largest rate boundary allowed by the existing clock-offset range is not
considered identified; the command fails without writing an output rather than
silently substituting the static model.

The same fail-closed rule applies elsewhere in the production pipeline. An
offset optimum at the configured search boundary is rejected, and a full
optical run whose motion series cannot be scaled to the nonzero ride average
fails instead of emitting fabricated constant-speed records.

The baseline method assumes that the Garmin zero-speed intervals are genuine stops.
Camera rotation during a stop can raise its median motion, which is why the
quietest stop is used. Without an observed stop, the additive baseline is not
identifiable and no subtraction is performed.

Scalar ranking, clock-offset search, stop-baseline estimation, mean scaling,
and validation metrics live in `speed_estimation.py` and are shared by the
production and public-validation pipelines.

## Demonstrations

`make_optical_flow_demo.py` exposes a reusable
`generate_optical_flow_demo(...)` function and a CLI. Production and demo code
share the exact preprocessing, Farneback, magnitude, ROI, and median operations.

```bash
python make_optical_flow_demo.py VIDEO.mp4 [OUTPUT_DIR] \
  --start 375 --duration 10 --sample-fps 4 --baseline 0.12
```

To locate and render the interval where two FIT outputs disagree most on
average, omit `--start` and provide both files:

```bash
python make_optical_flow_demo.py VIDEO.mp4 --duration 30 \
  --compare-fit OUTPUT-DRY.fit OUTPUT-FULL.fit
```

When `OUTPUT_DIR` is omitted, the CLI automatically creates a deterministic
folder under `~/Movies` using the duration and video-time range, for example
`optical-flow-demo-30s-1032-1102`. Passing a directory explicitly overrides the
automatic name.

After the full-frame source preview, stage 1 shows both downscaled, unblurred
grayscale inputs. Frame `N` contributes magenta, frame `N+1` contributes green,
and unchanged brightness appears gray. Full-frame Farneback vectors and
magnitudes follow. Only after magnitude calculation does the demo identify and
retain the central ROI used by the scalar median. The vector overlay is darkened
and desaturated, while each vector receives a contrasting black or white halo
selected from its Turbo-color luminance. The temporal overlay is explanatory
rather than an input: Farneback compares the two grayscale frames directly, and
an absolute-difference image would discard the direction needed to estimate
vectors. The optional baseline stage remains a spatial magnitude heatmap. The
subsequent ROI median and temporal speed series are scalar reductions and are
intended to be explained rather than presented as video transformations.

Every demo also produces `99_all_stages_spliced.mp4`. It divides the processed
observation frames as evenly as possible among all generated stages and joins
their corresponding timeline slices. For six stages across a six-second demo,
stage 1 supplies the first second, stage 2 the second, and so on; the video does
not restart when the displayed processing stage changes. The unprocessed source
preview remains separate from this stage-only splice.

## Public validation datasets

The repository includes end-to-end, modality-selective validators for two
public off-road datasets:

```bash
python tartandrive_validation.py
python sfu_mountain_validation.py --download
```

The TartanDrive runner downloads the selected camera, fused odometry, and four
wheel encoders, then reports estimator errors plus per-wheel disagreement.
The SFU runner selects forward camera, low-grade Garmin GPS, and wheel velocity
without requiring ROS. See `TARTANDRIVE_VALIDATION.md` and
`SFU_MOUNTAIN_VALIDATION.md` for dataset-specific limitations and commands.
