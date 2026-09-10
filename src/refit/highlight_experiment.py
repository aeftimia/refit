"""Run a repeatable multi-metric highlight comparison end to end."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .highlight_cli import build_timelines, discover_videos, duration_seconds
from .highlight_render import render_manifest
from .highlights import write_comparison_manifests


def local_date(value: str, timezone: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
        return parsed.replace(tzinfo=ZoneInfo(timezone))
    except (ValueError, KeyError) as exc:
        raise argparse.ArgumentTypeError(
            "since must be a YYYY-MM-DD date and timezone must be valid"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="refit-highlight-experiment")
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--since", required=True, help="inclusive local recording date")
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--duration", type=duration_seconds, required=True)
    parser.add_argument("--clip-duration", type=duration_seconds, default=10.0)
    parser.add_argument(
        "--order", choices=("chronological", "interesting"), default="interesting",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", default="hevc_videotoolbox")
    parser.add_argument("--height", type=int)
    return parser


def run(args: argparse.Namespace) -> dict:
    cutoff = local_date(args.since, args.timezone)
    videos = discover_videos(args.sources, recent_days=None)
    timelines = build_timelines(
        videos,
        args.fit_dir.expanduser().resolve(),
        recorded_since=cutoff,
        timezone=args.timezone,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"since_{args.since}"
    manifests = write_comparison_manifests(
        prefix,
        timelines,
        target_duration=args.duration,
        clip_duration=args.clip_duration,
        order=args.order,
    )
    exports = {}
    for mode, manifest in manifests.items():
        output = output_dir / f"since_{args.since}_{mode}.mp4"
        render_manifest(
            manifest, output, height=args.height, encoder=args.encoder,
        )
        exports[mode] = output
    return {"manifests": manifests, "exports": exports}


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    for mode, output in result["exports"].items():
        print(f"Wrote {mode}: {output}")


if __name__ == "__main__":
    main()
