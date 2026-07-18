"""Shared optical-flow preprocessing and measurement primitives."""

import cv2
import numpy as np

FRAME_WIDTH = 640
BLUR_KERNEL = (5, 5)
ROI = (.10, .20, .90, .85)  # left, top, right, bottom fractions


def resize_frame(frame, width=FRAME_WIDTH):
    scale = min(1.0, width / frame.shape[1])
    return cv2.resize(frame, None, fx=scale, fy=scale)


def grayscale_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def blur_frame(gray):
    return cv2.GaussianBlur(gray, BLUR_KERNEL, 0)


def preprocess_frame(frame, width=FRAME_WIDTH):
    """Return every production preprocessing stage for inspection or flow."""
    resized = resize_frame(frame, width)
    gray = grayscale_frame(resized)
    blurred = blur_frame(gray)
    return resized, gray, blurred


def calculate_flow(previous, current):
    return cv2.calcOpticalFlowFarneback(previous, current, None, .5, 3, 15, 3, 5, 1.2, 0)


def flow_magnitudes(flow):
    return np.linalg.norm(flow, axis=2)


def roi_bounds(shape):
    height, width = shape[:2]
    left, top, right, bottom = ROI
    return int(left * width), int(top * height), int(right * width), int(bottom * height)


def roi_magnitudes(magnitudes):
    x0, y0, x1, y1 = roi_bounds(magnitudes.shape)
    return magnitudes[y0:y1, x0:x1]


def median_flow_magnitude(previous, current):
    flow = calculate_flow(previous, current)
    return float(np.median(roi_magnitudes(flow_magnitudes(flow))))
