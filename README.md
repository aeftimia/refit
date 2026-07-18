# GPX correction for Insta360 telemetry

This project aligns a video clock with a Strava GPX activity and writes an
Insta360-compatible FIT file. Full mode replaces the Strava-derived speed shape
with an optical-motion estimate while preserving Strava's distance and average
speed. Dry-run mode keeps the Strava-derived speed for comparison.

## Optical speed pipeline

1. Uniformly sample adjacent video-frame pairs.
2. Downscale each frame to 640 pixels wide, convert it to grayscale, and apply
   the shared preprocessing in `optical_flow_pipeline.py`.
3. Compute dense Farneback optical flow between source frames `N` and `N+1`.
   Sampling controls how often one of these adjacent-frame pairs is measured;
   it does not increase the time separating the two frames in a pair. For
   example, 4 Hz analysis of 60 fps video uses approximately `(0, 1)`,
   `(15, 16)`, `(30, 31)`, and so on.
4. Convert each per-pixel flow vector to magnitude.
5. Retain the central spatial ROI and reduce it to its median magnitude.
6. Temporally smooth the resulting scalar motion series.
7. Align raw optical motion with Strava-derived speed using a static clock-shift
   search and Spearman rank correlation.
8. Identify contiguous, nonzero-duration intervals where the aligned GPX reports
   exactly zero speed.
9. Calculate the median optical magnitude within each observed stop. Use the
   lowest interval median as the additive optical baseline. This selects the
   quietest observed stop without relying on the lowest individual frame.
10. If no stationary interval exists, use a baseline of exactly zero. The code
    does not assume the camera is stationary for any fixed fraction of a ride.
11. Subtract the baseline and clamp negative results to zero.
12. Scale the corrected optical series so its time integral equals the Strava
    distance over the video/GPX overlap.
13. Store the result as FIT `enhanced_speed`, alongside aligned coordinates,
    elevation, distance, and heart rate.

The baseline method assumes that the GPX zero-speed intervals are genuine stops.
Camera rotation during a stop can raise its median motion, which is why the
quietest stop is used. Without an observed stop, the additive baseline is not
identifiable and no subtraction is performed.

## Demonstrations

`make_optical_flow_demo.py` exposes a reusable
`generate_optical_flow_demo(...)` function and a CLI. Production and demo code
share the exact preprocessing, Farneback, magnitude, ROI, and median operations.

```bash
python make_optical_flow_demo.py VIDEO.mp4 [OUTPUT_DIR] \
  --start 375 --duration 10 --sample-fps 4 --baseline 0.12
```

When `OUTPUT_DIR` is omitted, the CLI automatically creates a deterministic
folder under `~/Movies` using the duration and video-time range, for example
`optical-flow-demo-30s-1032-1102`. Passing a directory explicitly overrides the
automatic name.

After the source preview, the demonstration starts with both downscaled,
grayscale Farneback inputs in a temporal overlay: frame `N` contributes magenta,
frame `N+1` contributes green, and unchanged brightness appears gray. A separate
downscale-only clip is omitted because this display encoding would make it
visually redundant with the grayscale stage. Blur changes both encoded inputs,
and the vector stage retains their blurred overlay as its background while
coloring every arrow with the same global Turbo magnitude scale used by the
following heatmap. In the vector stage the overlay is darkened and desaturated,
while each vector receives a contrasting black or white halo selected from its
color luminance. That separates the two visual encodings without changing
magnitude colors. The overlay is explanatory rather than an input: Farneback
compares the two grayscale frames directly, and an absolute-difference image
would discard the direction needed to estimate vectors. The optional baseline
stage remains a spatial magnitude heatmap. The subsequent ROI median and
temporal speed series are scalar reductions and are intended to be explained
rather than presented as video transformations.

Every demo also produces `99_all_stages_spliced.mp4`. It divides the processed
observation frames as evenly as possible among all generated stages and joins
their corresponding timeline slices. For six stages across a six-second demo,
stage 1 supplies the first second, stage 2 the second, and so on; the video does
not restart when the displayed processing stage changes. The unprocessed source
preview remains separate from this stage-only splice.
