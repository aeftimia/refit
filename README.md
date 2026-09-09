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
of \(\Delta t\) is represented as

\[
\Delta t = k + r, \qquad k \in \mathbb{Z}, \quad -0.5 \le r < 0.5,
\]

where \(k\) shifts FIT timestamps and \(r\) shifts only the speed samples used
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
   \([10\%,90\%]\times[20\%,85\%]\). There is no temporal smoothing.
5. Search only positive offsets from 0 to 45 s, retaining the interior offset
   that maximizes linear correlation with Garmin `gps_metadata` speed. A
   boundary optimum is rejected rather than exported.

### Camera geometry

The source video is treated as a rectilinear/pinhole view. For an image width
\(W\) and configured horizontal field of view \(\theta_h\), its effective focal
length in pixels is

\[
f = \frac{W-1}{2\tan(\theta_h/2)}, \qquad
\mathbf q(u,v) = \frac{((u-c_x)/f,\,(v-c_y)/f,\,1)}
{\lVert((u-c_x)/f,\,(v-c_y)/f,\,1)\rVert}.
\]

That is the usual [pinhole projection model](https://docs.opencv.org/doc/doxygen/html/d2/d48/group__d__projection.html),
written here as pixels mapped to unit viewing rays. The Ace Pro 2 Bike Mode
profile lives in `camera_profiles.json`; it currently uses a 120° effective
horizontal FOV. That is a working estimate for this exported footage, not a
claim that the camera's raw lens is a 120° rectilinear lens. Insta360 lists a
157° lens FOV and multiple export FOV/stabilization modes, including High
stabilization for mountain biking ([specification](https://store.insta360.com/hr/product/ace-pro-2?c=3611&from=homepage),
[stabilization guide](https://onlinemanual.insta360.com/acepro2/en-us/faq/functionality/stabilization)).
Replace the profile with a calibration when one is available.

Let \(\mathbf w=(\dot u,\dot v)\) be pixel flow and \(J(u,v)\) the area
scaling induced by \(\mathbf q\). The scalar used for alignment is the median
of the discrete surface divergence

\[
\operatorname{div}_{S^2}\mathbf w
= \frac{1}{J}\left[
  \frac{\partial(J\dot u)}{\partial u} +
  \frac{\partial(J\dot v)}{\partial v}
\right].
\]

This compensates for the changing solid angle represented by a pixel away from
the image center. A rigid rotation induces a divergence-free tangent field on
the viewing sphere; forward camera translation tends to create outward image
expansion. The connection is the divergence/curl split behind the
[Helmholtz–Hodge decomposition](https://people.math.ethz.ch/~struwe/Skripten/NonEvolProb-HS2017.pdf).
ReFit does **not** perform a Helmholtz decomposition: it uses divergence as a
principled rotation-resistant motion proxy. Depth variation, independently
moving objects, stabilization artifacts, and an imperfect FOV calibration can
still affect it.

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
