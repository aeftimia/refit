#!/usr/bin/env python3
"""Render two FIT speed streams over identical side-by-side video frames."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from fit_binary import FitBinary
from video_speed_fit import parse_tz, video_window


MPH_PER_MPS = 2.2369362921


def fit_speed_series(path):
    fit = FitBinary(path)
    dense = fit.gps_metadata_points()
    if dense:
        return (
            np.array([timestamp for _, timestamp, _ in dense], dtype=float),
            np.array([speed for _, _, speed in dense], dtype=float),
        )
    records = [
        point for point in fit.track_records() if point.enhanced_speed is not None
    ]
    if len(records) < 2:
        raise ValueError(f"FIT has too few speed records: {path}")
    return (
        np.array([point.timestamp / 1000 for point in records], dtype=float),
        np.array([point.enhanced_speed for point in records], dtype=float),
    )


def video_timing(path):
    metadata = json.loads(subprocess.check_output([
        "exiftool", "-j", "-n", "-api", "QuickTimeUTC=1", "-Duration",
        "-CreateDate", "-MediaCreateDate", "-TrackCreateDate",
        "-DateTimeOriginal", "-TimeZone", "-OffsetTimeOriginal", str(path),
    ]))[0]
    start, end = video_window(metadata, parse_tz("UTC"))
    return start.timestamp(), (end - start).total_seconds()


def maximum_discrepancy_window(video, first_fit, second_fit, duration):
    video_start, video_duration = video_timing(video)
    first_t, first_v = fit_speed_series(first_fit)
    second_t, second_v = fit_speed_series(second_fit)
    overlap_start = max(video_start, first_t[0], second_t[0])
    overlap_end = min(
        video_start + video_duration, first_t[-1], second_t[-1],
    )
    if overlap_end - overlap_start < duration:
        raise ValueError("FIT/video overlap is shorter than the requested window")
    step = 0.05
    grid = np.arange(overlap_start, overlap_end + step / 2, step)
    difference = np.abs(
        np.interp(grid, first_t, first_v)
        - np.interp(grid, second_t, second_v)
    )
    sample_count = max(1, round(duration / step))
    rolling = np.convolve(
        difference, np.ones(sample_count) / sample_count, mode="valid",
    )
    index = int(np.argmax(rolling))
    start = float(grid[index] - video_start)
    return start, float(rolling[index])


def maximum_point_discrepancy_window(video, first_fit, second_fit, duration):
    """Center a window on the largest instantaneous absolute speed difference."""
    video_start, video_duration = video_timing(video)
    first_t, first_v = fit_speed_series(first_fit)
    second_t, second_v = fit_speed_series(second_fit)
    overlap_start = max(video_start, first_t[0], second_t[0])
    overlap_end = min(
        video_start + video_duration, first_t[-1], second_t[-1],
    )
    if overlap_end - overlap_start < duration:
        raise ValueError("FIT/video overlap is shorter than the requested window")
    step = 0.05
    grid = np.arange(overlap_start, overlap_end + step / 2, step)
    difference = np.abs(
        np.interp(grid, first_t, first_v)
        - np.interp(grid, second_t, second_v)
    )
    index = int(np.argmax(difference))
    event = float(grid[index])
    start = min(
        max(event - duration / 2, overlap_start),
        overlap_end - duration,
    )
    return (
        float(start - video_start),
        float(event - video_start),
        float(difference[index]),
    )


def default_output(video, start, duration):
    def slug(seconds):
        whole = max(0, round(seconds))
        return f"{whole // 60:02d}{whole % 60:02d}"

    return (
        Path.home() / "Movies"
        / (
            f"{Path(video).stem}-affine-vs-optical-"
            f"{duration:g}s-{slug(start)}-{slug(start + duration)}.mp4"
        )
    )


def put_text(frame, text, origin, scale, color, thickness=2):
    cv2.putText(
        frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
        thickness + 4, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
        thickness, cv2.LINE_AA,
    )


def panel(frame, label, speed, color, video_second):
    result = frame.copy()
    overlay = result.copy()
    cv2.rectangle(overlay, (0, 0), (result.shape[1], 112), (0, 0, 0), -1)
    cv2.rectangle(
        overlay, (0, result.shape[0] - 48),
        (result.shape[1], result.shape[0]), (0, 0, 0), -1,
    )
    cv2.addWeighted(overlay, 0.68, result, 0.32, 0, result)
    put_text(result, label, (24, 38), 0.78, color, 2)
    put_text(
        result, f"{speed * MPH_PER_MPS:5.1f} mph", (24, 92), 1.35,
        (255, 255, 255), 3,
    )
    minutes = int(video_second // 60)
    seconds = video_second - minutes * 60
    put_text(
        result, f"video {minutes:02d}:{seconds:05.2f}",
        (24, result.shape[0] - 16), 0.64, (230, 230, 230), 1,
    )
    return result


def combined_panel(frame, labels, speeds, video_second):
    result = frame.copy()
    overlay = result.copy()
    cv2.rectangle(overlay, (0, 0), (result.shape[1], 122), (0, 0, 0), -1)
    cv2.rectangle(
        overlay, (0, result.shape[0] - 48),
        (result.shape[1], result.shape[0]), (0, 0, 0), -1,
    )
    cv2.addWeighted(overlay, 0.68, result, 0.32, 0, result)
    colors = ((255, 220, 40), (40, 170, 255))
    columns = (24, result.shape[1] // 2 + 24)
    for label_text, speed, color, x in zip(labels, speeds, colors, columns):
        put_text(result, label_text, (x, 36), 0.72, color, 2)
        put_text(
            result, f"{speed * MPH_PER_MPS:5.1f} mph", (x, 94), 1.35,
            (255, 255, 255), 3,
        )
    delta = abs(speeds[0] - speeds[1]) * MPH_PER_MPS
    put_text(
        result, f"absolute difference {delta:.1f} mph",
        (result.shape[1] - 385, result.shape[0] - 16),
        0.58, (255, 255, 255), 1,
    )
    minutes = int(video_second // 60)
    seconds = video_second - minutes * 60
    put_text(
        result, f"video {minutes:02d}:{seconds:05.2f}",
        (24, result.shape[0] - 16), 0.64, (230, 230, 230), 1,
    )
    return result


def render(
    video, first_fit, second_fit, output, start, duration, labels,
    layout="single",
):
    video_start, _ = video_timing(video)
    first_t, first_v = fit_speed_series(first_fit)
    second_t, second_v = fit_speed_series(second_fit)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("video reports an invalid frame rate")
    frame_count = round(duration * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(start * fps))
    if layout == "single":
        pane_width, pane_height, output_size = 1280, 720, (1280, 720)
    else:
        pane_width, pane_height, output_size = 960, 540, (1920, 540)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".temporary.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        output_size,
    )
    if not writer.isOpened():
        raise ValueError(f"cannot create temporary video: {temporary}")
    started = time.monotonic()
    try:
        for index in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"video ended after {index} comparison frames")
            frame = cv2.resize(
                frame, (pane_width, pane_height), interpolation=cv2.INTER_AREA,
            )
            relative = start + index / fps
            absolute = video_start + relative
            first_speed = float(np.interp(absolute, first_t, first_v))
            second_speed = float(np.interp(absolute, second_t, second_v))
            if layout == "single":
                comparison = combined_panel(
                    frame, labels, (first_speed, second_speed), relative,
                )
            else:
                left = panel(
                    frame, labels[0], first_speed, (255, 220, 40), relative,
                )
                right = panel(
                    frame, labels[1], second_speed, (40, 170, 255), relative,
                )
                put_text(
                    right,
                    f"delta {abs(first_speed - second_speed) * MPH_PER_MPS:.1f} mph",
                    (pane_width - 235, 38), 0.62, (255, 255, 255), 1,
                )
                comparison = np.hstack((left, right))
            writer.write(comparison)
            if index == frame_count - 1 or index % max(1, round(fps)) == 0:
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = (index + 1) / elapsed
                eta = (frame_count - index - 1) / rate
                print(
                    f"\rRender {100 * (index + 1) / frame_count:5.1f}% "
                    f"ETA {eta:4.1f}s",
                    end="\n" if index == frame_count - 1 else "", flush=True,
                )
    finally:
        writer.release()
        cap.release()
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(temporary), "-an", "-c:v", "libx264", "-preset", "fast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-y", str(output),
    ], check=True)
    temporary.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("first_fit")
    parser.add_argument("second_fit")
    parser.add_argument("-o", "--output")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--start", type=float)
    parser.add_argument(
        "--selection", choices=("max-point", "max-mean"), default="max-point",
    )
    parser.add_argument(
        "--layout", choices=("single", "side-by-side"), default="single",
    )
    parser.add_argument(
        "--labels", nargs=2, default=("AFFINE GARMIN", "OPTICAL ESTIMATE"),
    )
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.start is None:
        if args.selection == "max-point":
            args.start, event, difference = maximum_point_discrepancy_window(
                args.video, args.first_fit, args.second_fit, args.duration,
            )
            print(
                f"Maximum point discrepancy occurs at {event:.3f}s; "
                f"{difference * MPH_PER_MPS:.2f} mph absolute difference",
            )
        else:
            args.start, mean_difference = maximum_discrepancy_window(
                args.video, args.first_fit, args.second_fit, args.duration,
            )
            print(
                f"Maximum {args.duration:g}s mean discrepancy begins at "
                f"{args.start:.3f}s "
                f"({mean_difference * MPH_PER_MPS:.2f} mph mean)",
            )
    output = Path(args.output) if args.output else default_output(
        args.video, args.start, args.duration,
    )
    render(
        args.video, args.first_fit, args.second_fit, output, args.start,
        args.duration, args.labels, args.layout,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
