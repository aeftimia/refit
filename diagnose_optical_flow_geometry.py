#!/usr/bin/env python3
"""Compare optical-flow geometries across a known stopped video interval.

This is deliberately a measurement diagnostic, not an alignment search.  It
uses the production resize, grayscale, Farneback, and ROI exactly as-is, then
reports raw-image and ray-space robust reductions of each same vector field.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from refit.optical_flow_pipeline import calculate_flow, flow_measurements, load_camera_profile


def observations(video, start, end, sample_fps, horizontal_fov_degrees):
    """Decode adjacent production-resolution pairs without materializing 4K frames."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open {video}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Video reports an invalid frame rate")
    if sample_fps <= 0 or sample_fps > source_fps / 2:
        raise ValueError(f"sample_fps must be in (0, {source_fps / 2:g}]")

    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    width = 640
    height = max(2, round(source_height * width / source_width / 2) * 2)
    step = max(2, round(source_fps / sample_fps))
    target_offsets = np.arange(0, end - start, step / source_fps)
    frame_bytes = width * height

    def decode_pair(offset):
        # This matches the production parallel sampler: seek directly to each
        # sample, decode two adjacent source frames, and downscale before flow.
        command = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start + offset:.6f}", "-i", str(video), "-frames:v", "2",
            "-map", "0:v:0", "-an",
            "-vf", f"scale={width}:{height},format=gray",
            "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode or len(result.stdout) < 2 * frame_bytes:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"FFmpeg sample at {start + offset:.3f}s failed: {error}")
        return np.frombuffer(result.stdout[:2 * frame_bytes], dtype=np.uint8).reshape(2, height, width)

    with ThreadPoolExecutor(max_workers=min(4, len(target_offsets))) as executor:
        pairs = list(executor.map(decode_pair, target_offsets))
    rows = []
    for offset, (first_gray, second_gray) in zip(target_offsets, pairs):
        measurement = flow_measurements(
            calculate_flow(first_gray, second_gray), horizontal_fov_degrees,
        )
        rows.append({"video_seconds": start + offset, **measurement})
    return rows


def write_svg(rows, output, start, stop_start, stop_end, end):
    """Write a dependency-free time-series plot with independently scaled rows."""
    width, height = 1100, 920
    left, right, top, bottom = 84, 30, 44, 38
    names = ("magnitude", "radial", "divergence", "spherical_divergence", "spherical_forward")
    colors = {
        "magnitude": "#277da1", "radial": "#2a9d8f", "divergence": "#e76f51",
        "spherical_divergence": "#7b2cbf", "spherical_forward": "#d00000",
    }
    plot_height = (height - top - bottom) / len(names)
    x = lambda t: left + (t - start) / (end - start) * (width - left - right)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="{x(stop_start):.1f}" y="{top}" width="{x(stop_end)-x(stop_start):.1f}" height="{height-top-bottom}" fill="#f6d365" opacity=".32"/>',
        '<style>text { font: 15px system-ui, sans-serif; fill: #202124 } .small { font-size: 12px }</style>',
        f'<text x="{left}" y="24">Optical-flow geometry diagnostic — shaded: known video stop</text>',
    ]
    for index, name in enumerate(names):
        values = np.array([row[name] for row in rows])
        lo, hi = float(values.min()), float(values.max())
        if hi == lo:
            lo, hi = lo - 1, hi + 1
        padding = (hi - lo) * .08
        lo, hi = lo - padding, hi + padding
        y0 = top + index * plot_height
        y = lambda value: y0 + plot_height - (value - lo) / (hi - lo) * plot_height
        points = " ".join(f"{x(row['video_seconds']):.1f},{y(row[name]):.1f}" for row in rows)
        parts.extend((
            f'<rect x="{left}" y="{y0:.1f}" width="{width-left-right}" height="{plot_height:.1f}" fill="none" stroke="#777"/>',
            f'<polyline points="{points}" fill="none" stroke="{colors[name]}" stroke-width="1.5"/>',
            f'<text x="8" y="{y0 + 22:.1f}" fill="{colors[name]}">{name}</text>',
            f'<text class="small" x="8" y="{y0 + 40:.1f}">{hi:.4g}</text>',
            f'<text class="small" x="8" y="{y0 + plot_height - 4:.1f}">{lo:.4g}</text>',
        ))
    for seconds in range(int(np.ceil(start)), int(np.floor(end)) + 1, 5):
        parts.append(f'<text class="small" x="{x(seconds)-8:.1f}" y="{height-12}">{seconds}s</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", type=float, required=True, help="video seconds")
    parser.add_argument("--stop-start", type=float, required=True, help="video seconds")
    parser.add_argument("--stop-end", type=float, required=True, help="video seconds")
    parser.add_argument("--end", type=float, required=True, help="video seconds")
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--camera-profile", default="insta360_ace_pro_2_bike_mode")
    parser.add_argument("--camera-profiles", type=Path)
    parser.add_argument("--output", type=Path, default=Path("optical-flow-geometry-diagnostic"))
    args = parser.parse_args()
    if not args.start <= args.stop_start < args.stop_end <= args.end:
        parser.error("Require start <= stop-start < stop-end <= end")
    args.output.mkdir(parents=True, exist_ok=True)
    profile = load_camera_profile(args.camera_profile, args.camera_profiles)
    print(
        f"Camera profile {args.camera_profile}: {profile['projection']}, "
        f"{profile['horizontal_fov_degrees']:g}° horizontal FOV"
    )
    rows = observations(
        args.video, args.start, args.end, args.sample_fps,
        float(profile["horizontal_fov_degrees"]),
    )
    if not rows:
        raise SystemExit("No frame pairs decoded")
    with (args.output / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_svg(rows, args.output / "measurements.svg", args.start, args.stop_start, args.stop_end, args.end)
    for name in ("magnitude", "radial", "divergence", "spherical_divergence", "spherical_forward"):
        stopped = [r[name] for r in rows if args.stop_start <= r["video_seconds"] <= args.stop_end]
        moving = [r[name] for r in rows if r["video_seconds"] < args.stop_start or r["video_seconds"] > args.stop_end]
        print(f"{name:10} moving median {np.median(moving): .6g} | stopped median {np.median(stopped): .6g}")
    print(f"Wrote {args.output / 'measurements.csv'} and {args.output / 'measurements.svg'}")


if __name__ == "__main__":
    main()
