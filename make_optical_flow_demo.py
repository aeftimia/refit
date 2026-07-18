#!/usr/bin/env python3
"""Render short videos illustrating the optical-flow preprocessing pipeline."""

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from optical_flow_pipeline import (
    BLUR_KERNEL, calculate_flow, flow_magnitudes, preprocess_frame,
    roi_bounds,
)
from fit_binary import FitBinary
from video_speed_fit import parse_tz, video_window


def writer(path, fps, size, color=True):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size, color)
    if not out.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    return out


def label(frame, text):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = .62
    text_width = cv2.getTextSize(text, font, scale, 1)[0][0]
    if text_width > frame.shape[1] - 20:
        scale *= (frame.shape[1] - 20) / text_width
    cv2.putText(frame, text, (10, 24), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def temporal_pair_overlay(previous, current):
    """Encode two grayscale frames as magenta/green temporal evidence."""
    return np.dstack((previous, current, previous))


def default_demo_output_dir(start, duration):
    """Return a stable Movies folder name derived from the video time range."""
    def timestamp_slug(seconds):
        whole_seconds = max(0, int(seconds))
        return f"{whole_seconds // 60:02d}{whole_seconds % 60:02d}"

    duration_slug = f"{duration:g}".replace(".", "p")
    range_slug = f"{timestamp_slug(start)}-{timestamp_slug(start + duration)}"
    return Path.home() / "Movies" / f"optical-flow-demo-{duration_slug}s-{range_slug}"


def highest_fit_discrepancy(video, dry_fit, full_fit, duration):
    """Locate the video interval with highest mean absolute FIT speed difference."""
    def series(path):
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
        return (
            np.array([point.timestamp / 1000 for point in records], dtype=float),
            np.array([point.enhanced_speed for point in records], dtype=float),
        )

    metadata = json.loads(subprocess.check_output([
        "exiftool", "-j", "-n", "-api", "QuickTimeUTC=1", "-Duration",
        "-CreateDate", "-MediaCreateDate", "-TrackCreateDate", "-DateTimeOriginal",
        "-TimeZone", "-OffsetTimeOriginal", str(video),
    ]))[0]
    video_start, _ = video_window(metadata, parse_tz("UTC"))
    dry_t, dry_v = series(dry_fit)
    full_t, full_v = series(full_fit)
    first = max(dry_t[0], full_t[0], video_start.timestamp())
    last = min(dry_t[-1], full_t[-1])
    step = .25
    grid = np.arange(first, last + step / 2, step)
    difference = np.abs(np.interp(grid, dry_t, dry_v) - np.interp(grid, full_t, full_v))
    samples = max(1, round(duration / step))
    if len(difference) < samples:
        raise ValueError("FIT/video overlap is shorter than the requested discrepancy window")
    rolling = np.convolve(difference, np.ones(samples) / samples, mode="valid")
    index = int(np.argmax(rolling))
    start = float(grid[index] - video_start.timestamp())
    print(
        f"Highest {duration:g}s mean speed discrepancy starts at video {start:.3f}s "
        f"(mean {rolling[index]:.3f} m/s)"
    )
    return start


def generate_optical_flow_demo(video, output_dir, start, duration, sample_fps=4.0, baseline=None):
    """Generate production-faithful stage clips and one timeline-continuous splice."""
    output = Path(output_dir) if output_dir else default_demo_output_dir(start, duration)
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
    if sample_fps <= 0 or sample_fps > source_fps / 2:
        raise ValueError(
            f"sample_fps must be in (0, {source_fps / 2:g}] so pairs do not overlap"
        )
    target_frames = np.rint(
        np.arange(start, start + duration, 1.0 / sample_fps) * source_fps
    ).astype(int)
    target_frames = np.unique(target_frames)
    target_frames = target_frames[target_frames + 1 < end_frame]
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    observations = []
    frame_number = start_frame
    for target_frame in target_frames:
        while frame_number < target_frame:
            if not cap.grab():
                break
            frame_number += 1
        if frame_number != target_frame:
            break
        first_frame_number = target_frame
        ok, first = cap.read()
        if not ok:
            break
        frame_number += 1
        ok, second = cap.read()
        if not ok:
            break
        frame_number += 1
        _, first_gray, first_blurred = preprocess_frame(first)
        _, gray, blurred = preprocess_frame(second)
        flow = calculate_flow(first_blurred, blurred)
        magnitude = flow_magnitudes(flow)
        observations.append((
            first_gray, gray, first_blurred, blurred, flow, magnitude,
            first_frame_number,
        ))
    cap.release()
    if not observations:
        raise RuntimeError("No observations decoded")

    h, w = observations[0][0].shape
    size = (w, h)
    magnitudes = np.concatenate([x[5].ravel() for x in observations])
    display_ceiling = max(float(np.percentile(magnitudes, 99)), 1e-6)
    paths = {
        "gray": output / "01_grayscale_pair_overlay.mp4",
        "blur": output / "02_blurred_pair_overlay.mp4",
        "vectors": output / "03_farneback_flow_vectors.mp4",
        "magnitude": output / "04_flow_magnitude.mp4",
        "roi": output / "05_roi_used_for_median.mp4",
    }
    if baseline is not None:
        paths["baseline"] = output / "06_baseline_subtracted_magnitude.mp4"
    for legacy_name in (
        "01_downscaled_640.mp4",
        "01_downscaled_pair_overlay.mp4",
        "02_grayscale.mp4",
        "02_grayscale_pair_overlay.mp4",
        "03_gaussian_blur_5x5.mp4",
        "03_blurred_pair_overlay.mp4",
        "03_farneback_flow_vectors.mp4",
        "04_farneback_flow_vectors.mp4",
        "04_consecutive_frame_overlay.mp4",
        "04_consecutive_frame_pair_and_difference.mp4",
        "04_flow_magnitude.mp4",
        "05_flow_magnitude.mp4",
        "05_farneback_flow_vectors.mp4",
        "05_roi_used_for_median.mp4",
        "06_flow_magnitude.mp4",
        "06_roi_used_for_median.mp4",
        "06_roi_median_signal.mp4",
        "06_baseline_subtracted_magnitude.mp4",
        "07_baseline_subtracted_magnitude.mp4",
        "07_roi_used_for_median.mp4",
        "08_baseline_subtracted_magnitude.mp4",
        "99_all_stages_spliced.mp4",
    ):
        legacy_path = output / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()
    outputs = {name: writer(path, sample_fps, size) for name, path in paths.items()}

    x0, y0, x1, y1 = roi_bounds((h, w))
    for (first_gray, gray, first_blurred, blurred, flow, magnitude,
         first_frame_number) in observations:
        timestamp = first_frame_number / source_fps
        suffix = f"video {timestamp // 60:02.0f}:{timestamp % 60:05.2f}"
        # Temporal color is a display encoding applied consistently to both
        # production inputs. It is not an array consumed by Farneback.
        gray_pair = temporal_pair_overlay(first_gray, gray)
        pair_overlay = temporal_pair_overlay(first_blurred, blurred)
        timing = f"magenta=N, green=N+1 (+{1000 / source_fps:.2f} ms)"
        outputs["gray"].write(label(
            gray_pair, f"1. Both frames grayscale at {w}x{h} | {timing} | {suffix}",
        ))
        outputs["blur"].write(label(
            pair_overlay.copy(),
            f"2. Both frames Gaussian-blurred {BLUR_KERNEL[0]}x{BLUR_KERNEL[1]} | overlay retained | {suffix}",
        ))

        normalized = np.clip(magnitude / display_ceiling * 255, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        # Turbo spans most hues, so hue choice alone cannot separate every arrow
        # from a colored background. Retain the temporal overlay at lower
        # saturation/value, then outline each arrow adaptively below.
        pair_gray = cv2.cvtColor(pair_overlay, cv2.COLOR_BGR2GRAY)
        vector_frame = cv2.addWeighted(
            cv2.cvtColor(pair_gray, cv2.COLOR_GRAY2BGR), .45,
            pair_overlay, .25,
            0,
        )
        spacing = 32
        arrow_scale = 8
        for y in range(spacing // 2, h, spacing):
            for x in range(spacing // 2, w, spacing):
                dx, dy = flow[y, x]
                end = (round(x + arrow_scale * dx), round(y + arrow_scale * dy))
                color = tuple(int(channel) for channel in heatmap[y, x])
                blue, green, red = color
                luminance = .114 * blue + .587 * green + .299 * red
                halo = (255, 255, 255) if luminance < 110 else (0, 0, 0)
                cv2.arrowedLine(
                    vector_frame, (x, y), end, halo, 4, cv2.LINE_AA,
                    tipLength=.3,
                )
                cv2.arrowedLine(
                    vector_frame, (x, y), end, color, 2, cv2.LINE_AA,
                    tipLength=.3,
                )
        outputs["vectors"].write(label(
            vector_frame,
            f"3. Vectors: direction=arrow, magnitude=Turbo color | {suffix}",
        ))

        outputs["magnitude"].write(label(heatmap.copy(), f"4. Flow magnitude (global 99th-percentile scale) | {suffix}"))

        roi_frame = (heatmap * .22).astype(np.uint8)
        roi_frame[y0:y1, x0:x1] = heatmap[y0:y1, x0:x1]
        cv2.rectangle(roi_frame, (x0, y0), (x1, y1), (255, 255, 255), 2)
        outputs["roi"].write(label(roi_frame, f"5. Spatial ROI supplied to median reduction | {suffix}"))
        if baseline is not None:
            corrected = np.maximum(magnitude - baseline, 0.0)
            corrected_normalized = np.clip(corrected / display_ceiling * 255, 0, 255).astype(np.uint8)
            corrected_heatmap = cv2.applyColorMap(corrected_normalized, cv2.COLORMAP_TURBO)
            corrected_roi = (corrected_heatmap * .22).astype(np.uint8)
            corrected_roi[y0:y1, x0:x1] = corrected_heatmap[y0:y1, x0:x1]
            cv2.rectangle(corrected_roi, (x0, y0), (x1, y1), (255, 255, 255), 2)
            outputs["baseline"].write(label(
                corrected_roi,
                f"6. ROI magnitude after subtracting baseline {baseline:.4f} px/frame | {suffix}",
            ))

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

    # Preserve the underlying video timeline while changing stages: stage i
    # contributes the corresponding i-th slice, rather than every stage
    # restarting at the beginning of the requested interval.
    stage_paths = list(paths.values())
    stage_count = len(stage_paths)
    frame_count = len(observations)
    if frame_count < stage_count:
        raise ValueError(
            f"Demo has only {frame_count} observations for {stage_count} stages; "
            "increase duration or sample rate"
        )
    boundaries = [round(i * frame_count / stage_count) for i in range(stage_count + 1)]
    spliced_path = output / "99_all_stages_spliced.mp4"
    spliced_temporary = output / "99_all_stages_spliced.mp4v.mp4"
    spliced_writer = writer(spliced_temporary, sample_fps, size)
    written = 0
    for path, first_frame, last_frame in zip(
        stage_paths, boundaries, boundaries[1:]
    ):
        stage_capture = cv2.VideoCapture(str(path))
        if not stage_capture.isOpened():
            raise RuntimeError(f"Could not reopen stage for splicing: {path}")
        for frame_index in range(last_frame):
            ok, frame = stage_capture.read()
            if not ok:
                stage_capture.release()
                raise RuntimeError(
                    f"Stage {path.name} ended before frame {last_frame}"
                )
            if frame_index >= first_frame:
                spliced_writer.write(frame)
                written += 1
        stage_capture.release()
    spliced_writer.release()
    if written != frame_count:
        raise RuntimeError(f"Stage splice wrote {written} of {frame_count} frames")
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(spliced_temporary), "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-y", str(spliced_path),
    ], check=True)
    spliced_temporary.unlink()

    print(
        f"Wrote source + {len(outputs)} processed clips + 1 stage splice "
        f"with {len(observations)} observations to {output}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument(
        "output_dir", nargs="?",
        help="default: ~/Movies/optical-flow-demo-{duration}s-{start}-{end}",
    )
    parser.add_argument("--start", type=float)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--baseline", type=float)
    parser.add_argument(
        "--compare-fit", nargs=2, metavar=("DRY.fit", "FULL.fit"),
        help="automatically use the interval with highest mean speed discrepancy",
    )
    args = parser.parse_args()
    if args.compare_fit:
        if args.start is not None:
            parser.error("--start and --compare-fit are mutually exclusive")
        args.start = highest_fit_discrepancy(
            args.video, args.compare_fit[0], args.compare_fit[1], args.duration
        )
    elif args.start is None:
        args.start = 375.0
    generate_optical_flow_demo(
        args.video, args.output_dir, args.start, args.duration, args.sample_fps, args.baseline
    )


if __name__ == "__main__":
    main()
