# GPX correction for Insta360 telemetry

This project aligns a video clock with a Strava GPX activity and writes an
Insta360-compatible FIT file. Full mode replaces the Strava-derived speed shape
with an optical-motion estimate while preserving Strava's distance and average
speed. Dry-run mode keeps the Strava-derived speed for comparison.

## Optical speed pipeline

1. Uniformly sample adjacent video-frame pairs.
2. Downscale each frame to 640 pixels wide, convert it to grayscale, and apply
   the shared preprocessing in `optical_flow_pipeline.py`.
3. Compute dense Farneback optical flow.
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
python make_optical_flow_demo.py VIDEO.mp4 OUTPUT_DIR \
  --start 375 --duration 10 --sample-fps 4 --baseline 0.12
```

The optional baseline stage remains a spatial magnitude heatmap. The subsequent
ROI median and temporal speed series are scalar reductions and are intended to
be explained rather than presented as video transformations.
