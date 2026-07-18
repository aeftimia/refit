# comma2k19 wheel-speed validation

This experiment evaluates the deployed dry and full methodologies against an
independent measurement that neither method can see during estimation.

## Experimental design

1. Remux the example's raw HEVC frames into MP4 without changing pixels.
2. Construct a Strava-like 1 Hz GPX using only u-blox GNSS coordinates,
   elevation, and timestamps. Do not include reported speed.
3. Run `insta360_video_speed_gpx.sh --dry-run` on the video and GPX.
4. Run the same CLI in full optical-flow mode on the same inputs.
5. After both estimates are complete, compare their FIT `enhanced_speed` values
   with the mean of the four CAN wheel-speed measurements.

Wheel speed is not used for clock alignment, baseline estimation, optical
normalization, or any other part of either estimator. Both methods use the same
automatically aligned video/GNSS inputs. The full method retains its production
behavior of normalizing optical motion to GNSS-derived distance.

By default the experiment uses comma2k19's authoritative shared boot-time clock
to correct the camera/GPX clock before either estimator runs. This prevents
nearly constant highway speed from producing different, weakly identified
correlation offsets for dry and full mode. `--clock-mode auto` is retained as a
separate end-to-end synchronization stress test; it must not be treated as a
fair estimator comparison when the two modes choose different offsets.

Metrics are evaluated on a uniform 20 Hz common timeline and include mean
absolute error, root mean squared error, signed bias, 95th-percentile absolute
error, and maximum absolute error. `comparison.csv` preserves the full aligned
series and `metrics.json` contains the summary.

## Running the included example

Download the comma2k19 repository and select its included example segment:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/commaai/comma2k19.git /tmp/comma2k19
git -C /tmp/comma2k19 sparse-checkout set Example_1

./venv/bin/python comma2k19_validation.py \
  '/tmp/comma2k19/Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40'
```

Generated artifacts are ignored by Git under `validation-output/`.

## Interpretation limits

This example is a fixed automotive camera on a highway, not a bicycle camera on
a wooded trail. It is a strong test of synchronization and whether the optical
shape improves on GNSS-derived speed, but a single segment cannot establish that
one pixels-per-frame calibration transfers across cameras, fields of view,
mounting heights, or scene-depth distributions. Evaluation on held-out segments
and both comma2k19 vehicles is required before making a general accuracy claim.

## Included example result

Using the authoritative shared camera/GNSS clock on example segment 40 produced:

| Method | MAE | Max absolute error | Pearson correlation |
|---|---:|---:|---:|
| Dry GNSS-derived speed | 0.268 m/s (0.600 mph) | 1.580 m/s (3.535 mph) | 0.994 |
| Full optical speed | 3.210 m/s (7.181 mph) | 8.834 m/s (19.761 mph) | 0.455 |

Dry mode wins both requested error metrics on this segment. Removing ten seconds
from each boundary does not reverse the result: dry MAE is 0.208 m/s and full
MAE is 3.008 m/s. This is therefore not primarily a convolution-edge artifact.

Inspection of the source video reveals independently moving cross traffic close
to the camera and windshield wipers repeatedly sweeping through the estimator's
ROI. These are valid counterexamples to the assumption that median dense image
motion is dominated by camera translation. They help explain why global
distance normalization corrects the optical mean but not its local speed shape.

As a separate stress test, independent automatic synchronization selected
−4.15 seconds for dry mode and +2.00 seconds for full mode even though the
authoritative clock mapping implies only about +0.19 seconds of metadata
rounding. The short urban segment has insufficiently distinctive speed variation
for unconstrained rank-correlation alignment. Those auto-synced errors are
preserved under `validation-output/comma2k19-auto-sync/` locally but are not a
fair comparison between the estimators.
