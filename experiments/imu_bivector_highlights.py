#!/usr/bin/env python3
"""Render an experimental 3D bivector highlight reel using Insta360 IMU data."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re

from refit.highlight_cli import build_timelines, discover_videos, duration_seconds, read_video_metadata
from refit.highlight_render import render_manifest
from refit.highlights import MotionTimeline, select_highlights
from refit.insta360_imu import accelerometer_samples, vertical_acceleration_envelope


def camera_proxy(video: Path, camera_dir: Path) -> Path:
    metadata = read_video_metadata(video)
    match = re.match(
        r"(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})",
        metadata["CreateDate"],
    )
    if not match:
        raise ValueError(f"cannot derive camera timestamp for {video}")
    stamp = "".join(match.groups()[:3]) + "_" + "".join(match.groups()[3:])
    candidates = sorted(camera_dir.glob(f"LRV_{stamp}_*.lrv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one camera proxy for {video.name}, found {len(candidates)}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--camera-dir", type=Path, required=True)
    parser.add_argument("--duration", type=duration_seconds, required=True)
    parser.add_argument("--clip-duration", type=duration_seconds, default=20)
    parser.add_argument("--max-clips-per-source", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int)
    parser.add_argument("--encoder", default="copy")
    args = parser.parse_args()

    videos = discover_videos(args.sources, recent_days=None)
    timelines = build_timelines(videos, args.fit_dir.resolve())
    augmented = []
    proxies = {}
    for timeline in timelines:
        video = Path(timeline.source)
        proxy = camera_proxy(video, args.camera_dir.resolve())
        imu_times, acceleration = accelerometer_samples(proxy)
        vertical = vertical_acceleration_envelope(
            imu_times, acceleration, timeline.motion.times
        )
        augmented.append(MotionTimeline(
            timeline.source,
            replace(timeline.motion, vertical_acceleration=vertical),
        ))
        proxies[str(video)] = str(proxy)

    clips = select_highlights(
        augmented,
        target_duration=args.duration,
        mode="bivector_3d",
        clip_duration=args.clip_duration,
        max_clips_per_source=args.max_clips_per_source,
        order="interesting",
    )
    manifest = {
        "score_mode": "bivector_3d",
        "score_description": "|v ∧ a| with GPS lateral and camera-IMU vertical acceleration",
        "target_duration_seconds": args.duration,
        "selected_duration_seconds": sum(clip.duration for clip in clips),
        "clip_duration_seconds": args.clip_duration,
        "max_clips_per_source": args.max_clips_per_source,
        "order": "interesting",
        "camera_proxies": proxies,
        "clips": [clip.as_dict() for clip in clips],
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    render_manifest(manifest_path, args.output, height=args.height, encoder=args.encoder)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
