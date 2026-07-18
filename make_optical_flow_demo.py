#!/usr/bin/env python3
"""Render short videos illustrating the optical-flow preprocessing pipeline."""

import argparse
from pathlib import Path
import subprocess

import cv2
import numpy as np

from optical_flow_pipeline import (
    BLUR_KERNEL, calculate_flow, flow_magnitudes, preprocess_frame,
    roi_bounds,
)


def writer(path, fps, size, color=True):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size, color)
    if not out.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    return out


def label(frame, text):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def generate_optical_flow_demo(video, output_dir, start, duration, sample_fps=4.0):
    """Generate reusable, production-faithful clips for each flow pipeline stage."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", str(start), "-t", str(duration), "-i", str(video),
        "-an", "-vf", "scale=1280:-2", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20",
        "-y", str(output / "00_source.mp4"),
    ], check=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = round(start * source_fps)
    end_frame = round((start + duration) * source_fps)
    sample_step = max(2, round(source_fps / sample_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    observations = []
    frame_number = start_frame
    while frame_number + 1 < end_frame:
        ok, first = cap.read()
        if not ok:
            break
        frame_number += 1
        ok, second = cap.read()
        if not ok:
            break
        frame_number += 1
        _, _, first_blurred = preprocess_frame(first)
        downscaled, gray, blurred = preprocess_frame(second)
        flow = calculate_flow(first_blurred, blurred)
        magnitude = flow_magnitudes(flow)
        observations.append((downscaled, gray, blurred, flow, magnitude))

        # Advance to the next uniformly sampled adjacent-frame pair.
        skip = sample_step - 2
        for _ in range(skip):
            if not cap.grab():
                break
            frame_number += 1
    cap.release()
    if not observations:
        raise RuntimeError("No observations decoded")

    h, w = observations[0][1].shape
    size = (w, h)
    magnitudes = np.concatenate([x[4].ravel() for x in observations])
    display_ceiling = max(float(np.percentile(magnitudes, 99)), 1e-6)
    paths = {
        "down": output / "01_downscaled_640.mp4",
        "gray": output / "02_grayscale.mp4",
        "blur": output / "03_gaussian_blur_5x5.mp4",
        "vectors": output / "04_farneback_flow_vectors.mp4",
        "magnitude": output / "05_flow_magnitude.mp4",
        "roi": output / "06_roi_used_for_median.mp4",
    }
    legacy_roi = output / "06_roi_median_signal.mp4"
    if legacy_roi.exists():
        legacy_roi.unlink()
    outputs = {name: writer(path, sample_fps, size) for name, path in paths.items()}

    x0, y0, x1, y1 = roi_bounds((h, w))
    for index, (downscaled, gray, blurred, flow, magnitude) in enumerate(observations):
        timestamp = start + index / sample_fps
        suffix = f"video {timestamp // 60:02.0f}:{timestamp % 60:05.2f}"
        outputs["down"].write(label(downscaled.copy(), f"1. Downscaled to {w}x{h} | {suffix}"))
        outputs["gray"].write(label(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), f"2. Grayscale | {suffix}"))
        outputs["blur"].write(label(cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR), f"3. Gaussian blur {BLUR_KERNEL[0]}x{BLUR_KERNEL[1]} | {suffix}"))

        vector_frame = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        spacing = 32
        arrow_scale = 8
        for y in range(spacing // 2, h, spacing):
            for x in range(spacing // 2, w, spacing):
                dx, dy = flow[y, x]
                end = (round(x + arrow_scale * dx), round(y + arrow_scale * dy))
                cv2.arrowedLine(vector_frame, (x, y), end, (0, 255, 0), 1, cv2.LINE_AA, tipLength=.3)
        outputs["vectors"].write(label(vector_frame, f"4. Farneback vectors (8x display scale) | {suffix}"))

        normalized = np.clip(magnitude / display_ceiling * 255, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        outputs["magnitude"].write(label(heatmap.copy(), f"5. Flow magnitude (global 99th-percentile scale) | {suffix}"))

        roi_frame = (heatmap * .22).astype(np.uint8)
        roi_frame[y0:y1, x0:x1] = heatmap[y0:y1, x0:x1]
        cv2.rectangle(roi_frame, (x0, y0), (x1, y1), (255, 255, 255), 2)
        outputs["roi"].write(label(roi_frame, f"6. Spatial ROI supplied to median reduction | {suffix}"))

    for out in outputs.values():
        out.release()

    # OpenCV's portable mp4v encoder produces green/static playback in some
    # macOS players. Finalize every demo as broadly compatible H.264/yuv420p.
    for path in paths.values():
        temporary = path.with_name(f"{path.stem}.h264.mp4")
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(path), "-an", "-c:v", "libx264", "-preset", "fast",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-y", str(temporary),
        ], check=True)
        path.unlink()
        temporary.replace(path)
    print(f"Wrote source + {len(outputs)} processed clips with {len(observations)} observations to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("output_dir")
    parser.add_argument("--start", type=float, default=375.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    args = parser.parse_args()
    generate_optical_flow_demo(
        args.video, args.output_dir, args.start, args.duration, args.sample_fps
    )


if __name__ == "__main__":
    main()
