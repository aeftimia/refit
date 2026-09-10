"""User-facing ReFit command."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .garmin_connect_fit import download_matching_fit
from .video_speed_fit import align_video


def read_video_metadata(video: Path) -> dict:
    """Read the timestamp and duration fields required for alignment."""
    if shutil.which("exiftool") is None:
        raise RuntimeError("exiftool is required and was not found on PATH")
    command = [
        "exiftool", "-j", "-n", "-api", "QuickTimeUTC=1",
        "-Duration", "-CreateDate", "-MediaCreateDate", "-TrackCreateDate",
        "-DateTimeOriginal", "-TimeZone", "-OffsetTimeOriginal", str(video),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    records = json.loads(result.stdout)
    if len(records) != 1:
        raise ValueError(f"exiftool returned {len(records)} metadata records")
    return records[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refit",
        description="Align Garmin telemetry to an action-camera video.",
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("fit", nargs="?", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--activity-id")
    parser.add_argument("--token-store", type=Path, default=Path("~/.garminconnect"))
    parser.add_argument("--cache-dir", type=Path, default=Path("~/.cache/refit/garmin"))
    parser.add_argument("--max-gap", type=float, default=45.0)
    parser.add_argument("--clock-offset", type=float, help=argparse.SUPPRESS)
    return parser


def run(args: argparse.Namespace) -> Path:
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    metadata = read_video_metadata(video)

    if args.fit is None:
        fit = download_matching_fit(
            metadata,
            args.token_store.expanduser(),
            args.cache_dir.expanduser(),
            activity_id=args.activity_id,
            max_gap=args.max_gap,
        )
    else:
        fit = args.fit.expanduser().resolve()
        if not fit.is_file():
            raise FileNotFoundError(f"Garmin FIT not found: {fit}")

    output = (
        args.output.expanduser().resolve()
        if args.output
        else Path.cwd() / f"{video.stem}_speed.fit"
    )
    align_video(
        str(video), str(fit), str(output), metadata,
        clock_offset=args.clock_offset,
    )
    return output


def main() -> None:
    try:
        output = run(build_parser().parse_args())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
