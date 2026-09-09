# ReFit — align Garmin telemetry to Insta360 video

ReFit finds the Garmin activity that overlaps an Insta360 MP4, estimates the
camera-to-Garmin clock offset from visual motion, and writes an
Insta360-compatible FIT sidecar. Import that FIT into Insta360 Studio's
[Stats dashboard](https://onlinemanual.insta360.com/studio/en-us/operation-guide/edit-function/dashboard-function)
to render speed, heart rate, route, and the rest of the original Garmin data.

The output is a minimally patched copy of the source FIT: it retains Garmin
messages, developer fields, positions, elevation, heart rate, events, and
device information. Only FIT timestamps and the `gps_metadata` speed samples
used by Studio are changed; checksums are recomputed.

## Use

```bash
cd ~/Downloads
bash /path/to/refit/insta360_video_speed_fit.sh VIDEO.mp4
```

The command downloads the matching Garmin FIT when none is provided, then
writes `VIDEO_speed.fit` in the working directory. On first use it asks for
Garmin Connect credentials and MFA; reusable authentication is stored locally.
To supply an already downloaded activity:

```bash
bash /path/to/refit/insta360_video_speed_fit.sh VIDEO.mp4 ACTIVITY.fit
```

The video creation timestamp must include a UTC offset. ReFit intentionally
refuses ambiguous camera time rather than inventing a timezone.

For a camera card containing recent footage, use the sequential batch helper:

```bash
bash /path/to/refit/batch_refit_recent.sh \
  /Volumes/Untitled/DCIM/Camera01 ~/Downloads
```

It processes MP4s modified in the last five days and writes one
`<video>_speed.fit` per video. The batch is sequential so camera-card I/O and
FFmpeg decoding remain predictable.

## What is aligned

The output FIT encodes the nearest whole-second timestamp correction. FIT
activity timestamps cannot represent fractional seconds, so the remaining phase
is applied by interpolating the `gps_metadata` speed stream. Thus a correction
of $\Delta t$ is represented as

$$
\Delta t = k + r, \qquad k \in \mathbb{Z}, \quad -0.5 \le r < 0.5,
$$

where $k$ shifts FIT timestamps and $r$ shifts only the speed samples used
by Studio. Garmin coordinates and record-level speed fields are untouched.

## Optical-motion alignment

ReFit is a timing estimator, not a monocular speedometer. Optical motion yields
a scale-free signal whose *shape* is aligned to Garmin speed with Pearson
correlation; Studio continues to display Garmin's speed scale.

1. Sample at most 1,000 pairs of **adjacent original-rate frames**, uniformly
   through the video. Sampling at 4 Hz from 60 fps, for example, measures
   `(0, 1)`, `(15, 16)`, `(30, 31)`, not frames four video frames apart.
2. Resize frames to 640 px wide, convert them to grayscale, and compute dense
   [Farnebäck optical flow](https://docs.opencv.org/3.4.19/dc/d6b/group__video__track.html)
   on the unblurred full frame.
3. Interpret the image-coordinate flow on the viewing sphere of a calibrated
   rectilinear camera model, then take its surface divergence.
4. Take the spatial median in the central measurement region
   $[10\%,90\%]\times[20\%,85\%]$. There is no temporal smoothing.
5. Search only positive offsets from 0 to 45 s, retaining the interior offset
   that maximizes linear correlation with Garmin `gps_metadata` speed. A
   boundary optimum is rejected rather than exported.

### Camera geometry

The source video is treated as a rectilinear/pinhole view. ReFit converts the
configured horizontal FOV into the equivalent focal length in pixels:

$$
f = \frac{W-1}{2\tan(\theta/2)}.
$$

Here, $W$ is the frame width in pixels, $\theta$ is the configured horizontal
FOV in radians, and $f$ is the resulting focal length in pixels. For each
pixel, the code uses its offset from the frame center and $f$ to form a
unit-length viewing direction. That is the usual
[pinhole projection model](https://docs.opencv.org/doc/doxygen/html/d2/d48/group__d__projection.html).
The Ace Pro 2 Bike Mode
profile in `camera_profiles.json` uses **120° horizontal FOV**. This is the
only FOV used by the calculation, and is a working estimate of the *exported,
stabilized rectilinear video*.

Insta360's published **157° lens FOV** is diagonal and describes the physical
lens, not the horizontal FOV of this stabilized export. It must not be inserted
into `horizontal_fov_degrees`: even if an unmodified 16:9 rectilinear image
really had a 157° diagonal FOV, that would imply roughly 154° horizontal, not
157°. Export FOV selection and Bike Mode/High stabilization can crop or warp
the raw lens image further ([specification](https://store.insta360.com/hr/product/ace-pro-2?c=3611&from=homepage),
[stabilization guide](https://onlinemanual.insta360.com/acepro2/en-us/faq/functionality/stabilization)).
The MP4 does not expose an FOV metadata tag, so calibration of the exported
image—not the raw-lens marketing number—is the way to replace 120°.

For the next expression, $(u,v)$ means a pixel location, and
$\mathbf w=(\dot u,\dot v)$ is the optical-flow displacement there (in pixels
per frame). $J(u,v)$ is the local sphere area represented by one image pixel,
computed from the FOV conversion above; $S^2$ denotes the unit viewing sphere.
The scalar used for alignment is the median of the discrete surface divergence

$$
\mathrm{div}_{S^2}\mathbf w
= \frac{1}{J}\left[
  \frac{\partial(J\dot u)}{\partial u} +
  \frac{\partial(J\dot v)}{\partial v}
\right].
$$

This compensates for the changing solid angle represented by a pixel away from
the image center. A rigid rotation induces a divergence-free tangent field on
the viewing sphere; forward camera translation tends to create outward image
expansion. ReFit therefore uses divergence directly as a rotation-resistant
motion proxy. Depth variation, independently moving objects, stabilization
artifacts, and an imperfect FOV calibration can still affect it.

Gaussian blur and pre-flow cropping were tested and removed. Full-frame,
unblurred flow was both simpler and more accurate on the included validation;
the late ROI remains because it is a robust spatial reduction, not a
preprocessing shortcut.

## Demos

`make_optical_flow_demo.py` uses the same decode, preprocessing, Farnebäck,
spherical-divergence, and ROI operations as production. It creates visual
stages only for image-valued transformations; scalar reductions are labeled but
not padded into artificial video stages.

```bash
python make_optical_flow_demo.py VIDEO.mp4 [OUTPUT_DIR] \
  --start 375 --duration 10 --sample-fps 4
```

To choose the interval where two FIT outputs differ most:

```bash
python make_optical_flow_demo.py VIDEO.mp4 --duration 30 \
  --compare-fit DRY.fit OPTICAL.fit
```

When no output directory is given, the demo uses a deterministic folder under
`~/Movies`. `99_all_stages_spliced.mp4` shows each visual stage once over its
corresponding slice of the chosen interval; it does not restart the clip per
stage.

## Validation

The repository includes reproducible, modality-selective validators for two
off-road datasets:

```bash
python tartandrive_validation.py
python sfu_mountain_validation.py --download
```

See [TartanDrive details](TARTANDRIVE_VALIDATION.md) and
[SFU Mountain details](SFU_MOUNTAIN_VALIDATION.md). Those results evaluate the
timing/motion proxy under their stated sensor assumptions; they do not turn
monocular optical flow into independently calibrated ground-truth speed.
