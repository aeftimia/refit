"""Create comparable highlight manifests from aligned video/FIT pairs."""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .cli import read_video_metadata
from .fit_binary import FitBinary
from .highlights import (
    MotionTimeline,
    geometric_motion_from_gps,
    write_comparison_manifests,
)
from .video_speed_fit import video_window


def duration_seconds(value: str) -> float:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError("use seconds or a value such as 90s, 10m, or 1h")
    scale = {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2)]
    result = float(match.group(1)) * scale
    if result <= 0:
        raise argparse.ArgumentTypeError("duration must be positive")
    return result


def discover_videos(paths: list[Path], recent_days: float | None) -> list[Path]:
    cutoff = None if recent_days is None else time.time() - recent_days * 86400
    videos = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path.is_dir():
            candidates = (
                candidate for candidate in path.glob("*.mp4")
                if not candidate.name.startswith("._")
            )
        elif path.is_file() and path.suffix.lower() == ".mp4":
            candidates = [path]
        else:
            raise FileNotFoundError(f"video or directory not found: {path}")
        videos.extend(
            candidate for candidate in candidates
            if cutoff is None or candidate.stat().st_mtime >= cutoff
        )
    return sorted(set(videos))


def aligned_fit_path(
    video: Path,
    fit_dir: Path,
    metadata: dict | None = None,
    *,
    timezone: str = "America/New_York",
    tolerance_seconds: float = 2.0,
) -> Path:
    """Match a sidecar directly or by the camera timestamp in its filename."""
    direct = fit_dir / f"{video.stem}_speed.fit"
    if direct.is_file() or metadata is None:
        return direct
    recorded_at = video_window(metadata)[0]
    candidates = []
    for candidate in fit_dir.glob("*_speed.fit"):
        match = re.search(r"(\d{8})_(\d{6})", candidate.name)
        if not match:
            continue
        local_start = datetime.strptime(
            "_".join(match.groups()), "%Y%m%d_%H%M%S"
        ).replace(tzinfo=ZoneInfo(timezone))
        difference = abs((local_start - recorded_at).total_seconds())
        candidates.append((difference, candidate))
    if not candidates:
        return direct
    difference, closest = min(candidates, key=lambda item: item[0])
    return closest if difference <= tolerance_seconds else direct


def build_timelines(
    videos: list[Path],
    fit_dir: Path,
    *,
    smoothing_seconds: float,
    recorded_since: datetime | None = None,
    timezone: str = "America/New_York",
) -> list[MotionTimeline]:
    """Load chronological timelines, optionally filtering by recording time."""
    pairs = []
    missing = []
    for video in videos:
        metadata = read_video_metadata(video)
        recorded_at = video_window(metadata)[0]
        if recorded_since is not None and recorded_at < recorded_since:
            continue
        fit_path = aligned_fit_path(
            video, fit_dir, metadata, timezone=timezone,
        )
        if fit_path.is_file():
            pairs.append((recorded_at, video, fit_path, metadata))
        else:
            missing.append(fit_path)
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"aligned FIT sidecars not found:\n{formatted}")
    if not pairs:
        raise ValueError("no eligible recorded videos found")
    pairs.sort(key=lambda item: item[0])
    return [
        timeline_from_aligned_fit(
            video, fit_path, metadata,
            smoothing_seconds=smoothing_seconds,
        )
        for _, video, fit_path, metadata in pairs
    ]


def timeline_from_aligned_fit(
    video: Path,
    fit_path: Path,
    metadata: dict,
    *,
    smoothing_seconds: float,
) -> MotionTimeline:
    """Build video-relative geometric motion from an already aligned sidecar."""
    start, end = video_window(metadata)
    fit = FitBinary(fit_path)
    records = fit.track_records()
    if len(records) < 3:
        raise ValueError(f"FIT has too few positioned records: {fit_path}")
    record_times = np.array([record.timestamp / 1000 for record in records], dtype=float)
    lower = max(start.timestamp(), record_times[0])
    upper = min(end.timestamp(), record_times[-1])
    latitude = np.array([record.position_lat for record in records], dtype=float)
    record_longitude = np.unwrap(np.radians([
        record.position_long for record in records
    ]))
    longitude = np.degrees(record_longitude)

    dense = fit.gps_metadata_points()
    if len(dense) >= 2:
        speed_times = np.array([timestamp for _, timestamp, _ in dense])
        speed_values = np.array([speed for _, _, speed in dense])
        lower = max(lower, speed_times[0])
        upper = min(upper, speed_times[-1])
        speed = speed_values
    else:
        record_speed = np.array([
            record.enhanced_speed
            if record.enhanced_speed is not None else np.nan
            for record in records
        ])
        valid = np.isfinite(record_speed)
        speed_times = record_times[valid] if valid.sum() >= 2 else None
        speed = record_speed[valid] if valid.sum() >= 2 else None
        if speed_times is not None:
            lower = max(lower, speed_times[0])
            upper = min(upper, speed_times[-1])

    sample_times = np.arange(
        math.ceil(lower * 10) / 10,
        math.floor(upper * 10) / 10 + 0.05,
        0.1,
    )
    if len(sample_times) < 3:
        raise ValueError(f"video/FIT overlap is too short: {video}")
    epoch = start.timestamp()
    motion = geometric_motion_from_gps(
        record_times - epoch,
        latitude,
        longitude,
        speed=speed,
        speed_times=None if speed_times is None else speed_times - epoch,
        sample_times=sample_times - epoch,
        smoothing_seconds=smoothing_seconds,
    )
    return MotionTimeline(str(video), motion)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refit-highlights",
        description="Compare lateral, bivector, and total geometric highlight rankings.",
    )
    parser.add_argument("sources", nargs="+", type=Path, help="MP4 files or directories")
    parser.add_argument("--fit-dir", type=Path, default=Path.cwd())
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--recent-days", type=float)
    parser.add_argument("--duration", type=duration_seconds, required=True)
    parser.add_argument("--clip-duration", type=duration_seconds, default=10.0)
    parser.add_argument(
        "--smoothing", type=duration_seconds, default=1.0,
        help="velocity smoothing window (default: 1s)",
    )
    parser.add_argument(
        "--order", choices=("chronological", "interesting"),
        default="chronological",
    )
    parser.add_argument("--output-prefix", type=Path, default=Path("highlights"))
    return parser


def run(args: argparse.Namespace) -> dict:
    videos = discover_videos(args.sources, args.recent_days)
    if not videos:
        raise ValueError("no eligible MP4 videos found")
    fit_dir = args.fit_dir.expanduser().resolve()
    timelines = build_timelines(
        videos, fit_dir, smoothing_seconds=args.smoothing, timezone=args.timezone,
    )
    return write_comparison_manifests(
        args.output_prefix.expanduser().resolve(),
        timelines,
        target_duration=args.duration,
        clip_duration=args.clip_duration,
        order=args.order,
    )


def main() -> None:
    try:
        outputs = run(build_parser().parse_args())
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    for mode, output in outputs.items():
        print(f"Wrote {mode}: {output}")


if __name__ == "__main__":
    main()
