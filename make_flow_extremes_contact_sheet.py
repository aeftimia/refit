#!/usr/bin/env python3
"""Render low/high optical-flow moments with a candidate Garmin-speed label."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import timezone
from pathlib import Path

import cv2
import numpy as np

from fit_binary import FitBinary
from video_speed_fit import video_window


def video_start(video):
    metadata = json.loads(subprocess.check_output([
        "exiftool", "-j", "-n", "-api", "QuickTimeUTC=1",
        "-CreateDate", "-MediaCreateDate", "-TrackCreateDate", "-DateTimeOriginal",
        "-Duration", str(video),
    ]))[0]
    return video_window(metadata)[0].astimezone(timezone.utc).timestamp()


def frame_at(video, seconds, width=480):
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{seconds:.6f}", "-i", str(video), "-frames:v", "1",
        "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    image = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode video {seconds:.3f}s")
    return image


def label(image, heading, row, speed):
    h, w = image.shape[:2]
    result = cv2.copyMakeBorder(image, 0, 52, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(result, heading, (10, h + 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                (74, 222, 128), 1, cv2.LINE_AA)
    text = (
        f"video {float(row['video_seconds']):.1f}s | radial {float(row['radial']):.3f} "
        f"px/frame | Garmin {speed * 2.23694:.1f} mph"
    )
    cv2.putText(result, text, (10, h + 42), cv2.FONT_HERSHEY_SIMPLEX, .44,
                (255, 255, 255), 1, cv2.LINE_AA)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("fit", type=Path)
    parser.add_argument("--offset", type=float, required=True,
                        help="Garmin seconds after the MP4 clock")
    parser.add_argument("--count", type=int, default=6, help="frames per extreme")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.measurements.open()))
    if len(rows) < args.count * 2:
        parser.error("Measurement CSV has too few rows")
    selected = sorted(rows, key=lambda row: float(row["radial"]))[:args.count]
    selected += sorted(rows, key=lambda row: float(row["radial"]), reverse=True)[:args.count]
    points = FitBinary(args.fit).gps_metadata_points()
    times = np.array([timestamp for _, timestamp, _ in points])
    speeds = np.array([speed for _, _, speed in points])
    start = video_start(args.video)
    tiles = []
    for index, row in enumerate(selected):
        seconds = float(row["video_seconds"])
        speed = np.interp(start + seconds + args.offset, times, speeds)
        heading = "LOW radial expansion" if index < args.count else "HIGH radial expansion"
        tiles.append(label(frame_at(args.video, seconds), heading, row, speed))

    columns = 3
    tile_h, tile_w = tiles[0].shape[:2]
    canvas = np.zeros((tile_h * 4, tile_w * columns, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y, x = divmod(index, columns)
        canvas[y * tile_h:(y + 1) * tile_h, x * tile_w:(x + 1) * tile_w] = tile
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise ValueError(f"Could not write {args.output}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
