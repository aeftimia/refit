"""Shared optical-flow preprocessing and measurement primitives."""

import json
from pathlib import Path

import cv2
import numpy as np

FRAME_WIDTH = 640
ROI = (.10, .20, .90, .85)  # left, top, right, bottom fractions
DEFAULT_HORIZONTAL_FOV_DEGREES = 120.0


def load_camera_profile(name, profiles_path=None):
    """Load a swappable camera/lens specification from JSON."""
    path = Path(profiles_path) if profiles_path else Path(__file__).with_name("camera_profiles.json")
    profiles = json.loads(path.read_text())["profiles"]
    try:
        profile = profiles[name]
    except KeyError as exc:
        raise ValueError(f"Unknown camera profile {name!r}; available: {', '.join(profiles)}") from exc
    if profile.get("projection") != "rectilinear":
        raise ValueError(f"Profile {name!r} has unsupported projection {profile.get('projection')!r}")
    fov = float(profile["horizontal_fov_degrees"])
    if not 0 < fov < 180:
        raise ValueError(f"Profile {name!r} has invalid horizontal FOV {fov}")
    return profile


def resize_frame(frame, width=FRAME_WIDTH):
    scale = min(1.0, width / frame.shape[1])
    return cv2.resize(frame, None, fx=scale, fy=scale)


def grayscale_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def preprocess_frame(frame, width=FRAME_WIDTH):
    """Return the production downscale and grayscale stages."""
    resized = resize_frame(frame, width)
    return resized, grayscale_frame(resized)


def calculate_flow(previous, current):
    return cv2.calcOpticalFlowFarneback(
        previous, current, None, .5, 3, 15, 3, 5, 1.2, 0
    )


def flow_magnitudes(flow):
    return np.linalg.norm(flow, axis=2)


def roi_bounds(shape):
    height, width = shape[:2]
    left, top, right, bottom = ROI
    return int(left * width), int(top * height), int(right * width), int(bottom * height)


def roi_magnitudes(magnitudes):
    x0, y0, x1, y1 = roi_bounds(magnitudes.shape)
    return magnitudes[y0:y1, x0:x1]


def roi_values(values):
    """Return the production measurement ROI from a scalar or vector image."""
    x0, y0, x1, y1 = roi_bounds(values.shape)
    return values[y0:y1, x0:x1]


def signed_radial_flow(flow):
    """Project dense flow onto outward rays from the image centre.

    Forward camera motion has positive radial flow (image expansion); a pure
    rotation is tangential and therefore contributes approximately zero.  This
    is a diagnostic primitive for now, not part of production alignment.
    """
    height, width = flow.shape[:2]
    y, x = np.mgrid[:height, :width]
    x = x - (width - 1) / 2
    y = y - (height - 1) / 2
    radius = np.hypot(x, y)
    # The centre pixel has no defined radial direction.  Its value cannot
    # affect the median over the large measurement ROI, so define it as zero.
    radius[radius == 0] = 1
    return (flow[..., 0] * x + flow[..., 1] * y) / radius


def flow_divergence(flow):
    """Return local 2-D flow divergence in pixel displacement per pixel.

    Translation and rotation have zero divergence in the ideal image model;
    forward motion produces positive divergence (outward expansion).
    """
    d_horizontal_dx = np.gradient(flow[..., 0], axis=1)
    d_vertical_dy = np.gradient(flow[..., 1], axis=0)
    return d_horizontal_dx + d_vertical_dy


def spherical_geometry(shape, horizontal_fov_degrees=DEFAULT_HORIZONTAL_FOV_DEGREES):
    """Return rays, their pixel-coordinate tangent bases, and sphere area density."""
    height, width = shape[:2]
    focal = (width - 1) / (2 * np.tan(np.deg2rad(horizontal_fov_degrees) / 2))
    y, x = np.mgrid[:height, :width]
    raw = np.stack(((x - (width - 1) / 2) / focal,
                    (y - (height - 1) / 2) / focal,
                    np.ones_like(x)), axis=2).astype(float)
    raw_norm = np.linalg.norm(raw, axis=2, keepdims=True)
    rays = raw / raw_norm
    # d(normalize(raw))/dp = (I - q q^T) d(raw)/dp.
    x_basis = (np.dstack((np.ones_like(x), np.zeros_like(x), np.zeros_like(x)))
               - rays * rays[..., :1]) / (focal * raw_norm)
    y_basis = (np.dstack((np.zeros_like(x), np.ones_like(x), np.zeros_like(x)))
               - rays * rays[..., 1:2]) / (focal * raw_norm)
    area_density = np.linalg.norm(np.cross(x_basis, y_basis), axis=2)
    return rays, x_basis, y_basis, area_density


def spherical_flow(flow, horizontal_fov_degrees=DEFAULT_HORIZONTAL_FOV_DEGREES):
    """Map pixel/frame flow to tangent velocity on the rectilinear viewing sphere."""
    rays, x_basis, y_basis, _ = spherical_geometry(flow.shape, horizontal_fov_degrees)
    return rays, x_basis * flow[..., :1] + y_basis * flow[..., 1:2]


def spherical_divergence(flow, horizontal_fov_degrees=DEFAULT_HORIZONTAL_FOV_DEGREES):
    """Surface divergence; ideal rigid rotation is divergence-free on the sphere."""
    _, _, _, area_density = spherical_geometry(flow.shape, horizontal_fov_degrees)
    numerator = (np.gradient(area_density * flow[..., 0], axis=1)
                 + np.gradient(area_density * flow[..., 1], axis=0))
    return numerator / area_density


def spherical_forward_projection(flow, horizontal_fov_degrees=DEFAULT_HORIZONTAL_FOV_DEGREES):
    """Project ray-space flow onto the forward-translation template.

    This measures translation-like coherence only: monocular depth remains
    unknown, so it is not yet metric speed.
    """
    rays, tangent_flow = spherical_flow(flow, horizontal_fov_degrees)
    axis = np.zeros_like(rays)
    axis[..., 2] = 1
    toward_axis = axis - rays * rays[..., 2:3]
    norm = np.linalg.norm(toward_axis, axis=2)
    norm[norm == 0] = 1
    # A camera translating forward produces outward ray motion, -toward_axis.
    return -np.sum(tangent_flow * toward_axis, axis=2) / norm


def flow_measurements(flow, horizontal_fov_degrees=DEFAULT_HORIZONTAL_FOV_DEGREES):
    """Robust scalar reductions used to compare optical-flow geometries."""
    return {
        "magnitude": float(np.median(roi_values(flow_magnitudes(flow)))),
        "radial": float(np.median(roi_values(signed_radial_flow(flow)))),
        "divergence": float(np.median(roi_values(flow_divergence(flow)))),
        "spherical_divergence": float(np.median(roi_values(
            spherical_divergence(flow, horizontal_fov_degrees)
        ))),
        "spherical_forward": float(np.median(roi_values(
            spherical_forward_projection(flow, horizontal_fov_degrees)
        ))),
    }


def median_flow_magnitude(previous, current):
    """Compute unblurred full-frame flow, then reduce the measurement ROI."""
    flow = calculate_flow(previous, current)
    return float(np.median(roi_magnitudes(flow_magnitudes(flow))))
