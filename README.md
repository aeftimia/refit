# FIT speed correction for Insta360 telemetry

This project aligns a video clock with an original Garmin FIT activity and
writes an Insta360-compatible FIT file. The writer patches fixed-width FIT
timestamp and speed fields in the original binary and recalculates its CRCs;
every non-target byte is retained, including Garmin messages, developer fields,
GPS metadata, heart rate, elevation, events, and device information. The command
synchronizes timestamps while retaining Garmin's recorded speed and position.

The shortest workflow needs only the video. The first run prompts securely for
Garmin Connect credentials and MFA when required; later runs reuse refreshable
tokens stored in `~/.garminconnect`. The matching original FIT is cached under
`~/.cache/refit/garmin`, then passed through the same lossless transformation.
The launcher always uses the repository's `venv/bin/python`.

```bash
bash insta360_video_speed_fit.sh VIDEO.mp4
```

The default output is `<video-basename>_speed.fit` in the directory where the
command is run. For example, running against `/Volumes/Camera/VID_001.mp4` from
`~/Downloads` writes `~/Downloads/VID_001_speed.fit`.

To use a local Garmin FIT rather than downloading it:

```bash
bash insta360_video_speed_fit.sh VIDEO.mp4 GARMIN.fit
```

The MP4 creation timestamp must include a UTC offset. ReFit refuses to guess
a timezone when that metadata is incomplete.

Standard FIT timestamps have whole-second resolution. The synchronization
encodes the nearest timestamp shift, then applies any fractional residual by
interpolating Garmin's denser `gps_metadata` speed stream. Coordinates and
record-level Garmin speeds remain unchanged.

Automatic selection searches Garmin activities around the video date and picks
the activity with the greatest timeline overlap. The Garmin Connect downloader
uses Garmin's mobile authentication flow through
the third-party `garminconnect` package; it is separate from Garmin's official
Activity API, which requires Developer Program approval for a registered cloud
integration.

## Optical speed pipeline

Clock alignment uses up to 1,000 uniformly spaced adjacent-frame pairs across
the video, searching only positive 0–45 second camera-clock corrections. Every
confirmed correction has been in that direction; an optimum at either boundary
is rejected rather than exported.

1. Uniformly sample adjacent video-frame pairs at the selected analysis rate.
2. Downscale each frame to 640 pixels wide and convert it to grayscale.
3. Compute dense Farneback optical flow directly between the unblurred,
   full-frame grayscale source frames `N` and `N+1`.
   Sampling controls how often one of these adjacent-frame pairs is measured;
   it does not increase the time separating the two frames in a pair. For
   example, 4 Hz analysis of 60 fps video uses approximately `(0, 1)`,
   `(15, 16)`, `(30, 31)`, and so on.
4. Project the flow onto calibrated camera rays using the Ace Pro 2 Bike Mode
   profile in `camera_profiles.json`.
5. Compute spherical surface divergence. Ideal rigid camera rotation is
   divergence-free in this geometry; forward translation produces expansion.
6. Select the central spatial ROI and reduce its spherical divergence to a
   robust median scalar.
7. Preserve the resulting scalar motion series without temporal smoothing.
   Spatial median reduction already rejects pixel outliers, while an assumed
   temporal cutoff could erase genuine acceleration and braking.
8. Align raw optical motion with Garmin's approximately 1 Hz `gps_metadata`
   speed stream using a static clock-shift search and linear correlation. This
   retains the depth of low-motion valleys, so sustained stops can contribute
   proportionally to timing alignment.
9. Encode the clock correction in whole-second FIT timestamps. Interpolate the
   denser Garmin GPS-metadata speeds by the fractional remainder, preserving
   subsecond video alignment without changing the source video, position, or
   record-level Garmin speed.

Gaussian blur and padded early cropping were tested on TartanDrive and removed.
Among those preprocessing variants, unblurred full-frame flow produced the best
wheel-referenced MAE and correlation. Early cropping improved isolated flow
throughput but slightly worsened accuracy, changed the signal during large
motion, and had little effect on dry-run wall time because video seeking
dominates.

The later measurement ROI is separate and intentional: no pixels are discarded
before Farneback flow or magnitude calculation. Only the scalar median uses the
central region spanning 10%–90% of frame width and 20%–85% of frame height.
The demonstrations darken pixels outside that region to make the reduction
visible. Removing this late ROI also worsened the TartanDrive validation.

The production pipeline fails closed rather than silently substituting another
method: an offset optimum at either search boundary is rejected.

Clock-offset search and validation metrics live in `speed_estimation.py` and
are shared by the production and public-validation pipelines.

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
their corresponding timeline slices. For five stages across a five-second demo,
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
