#!/usr/bin/env python3
"""Find and download the Garmin Connect activity matching a video window."""

from __future__ import annotations

import argparse
import getpass
import io
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class ActivityWindow:
    activity_id: str
    name: str
    start: datetime
    end: datetime


def prompt_line(message: str) -> str:
    """Prompt interactively without contaminating stdout's machine-readable path."""
    print(message, end="", file=sys.stderr, flush=True)
    return sys.stdin.readline().strip()


def parse_garmin_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def activity_window(activity: dict) -> ActivityWindow:
    start_value = activity.get("startTimeGMT") or activity.get("startTimeLocal")
    if not start_value:
        raise ValueError("activity has no start time")
    start = parse_garmin_time(str(start_value))
    duration = activity.get("elapsedDuration", activity.get("duration", 0))
    duration = max(0.0, float(duration or 0))
    activity_id = activity.get("activityId")
    if activity_id is None:
        raise ValueError("activity has no activity ID")
    return ActivityWindow(
        str(activity_id),
        str(
            activity.get("activityName")
            or activity.get("activityType", {}).get("typeKey")
            or "activity"
        ),
        start,
        start + timedelta(seconds=duration),
    )


def interval_gap(
    a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime
) -> float:
    if a_end < b_start:
        return (b_start - a_end).total_seconds()
    if b_end < a_start:
        return (a_start - b_end).total_seconds()
    return 0.0


def overlap_seconds(
    a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime
) -> float:
    return max(0.0, (min(a_end, b_end) - max(a_start, b_start)).total_seconds())


def choose_activity(
    activities: list[dict],
    video_start: datetime,
    video_end: datetime,
    max_gap: float = 300.0,
) -> ActivityWindow:
    candidates = []
    for activity in activities:
        try:
            window = activity_window(activity)
        except (TypeError, ValueError):
            continue
        overlap = overlap_seconds(video_start, video_end, window.start, window.end)
        gap = interval_gap(video_start, video_end, window.start, window.end)
        start_delta = abs((window.start - video_start).total_seconds())
        # Prefer the activity containing the most video, then the nearest interval
        # and start. This naturally handles a video that ends after the activity.
        candidates.append(((-overlap, gap, start_delta), window))
    if not candidates:
        raise ValueError("Garmin Connect returned no activities with usable timestamps")
    candidates.sort(key=lambda item: item[0])
    score, selected = candidates[0]
    if score[1] > max_gap:
        raise ValueError(
            "No Garmin activity overlaps the video or falls within "
            f"{max_gap:g}s of it; nearest activity is {score[1]:.1f}s away"
        )
    return selected


def extract_fit(payload: bytes) -> bytes:
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".fit")]
            if not names:
                raise ValueError("Garmin activity archive contains no FIT file")
            # Garmin may include settings alongside the activity. Prefer the
            # largest FIT, which is the recorded activity stream.
            name = max(names, key=lambda item: archive.getinfo(item).file_size)
            payload = archive.read(name)
    if len(payload) < 14 or payload[8:12] != b".FIT":
        raise ValueError("Garmin download is not a valid FIT activity file")
    return payload


def connect(token_store: Path):
    try:
        from garminconnect import Garmin
        from garminconnect import GarminConnectAuthenticationError
    except ImportError as exc:
        raise SystemExit(
            "garminconnect is required for automatic downloads; "
            "install with: python3 -m pip install -r requirements.txt"
        ) from exc

    prompt_mfa = lambda: prompt_line("Garmin MFA code: ")
    client = Garmin(prompt_mfa=prompt_mfa)
    try:
        client.login(str(token_store))
        return client
    except GarminConnectAuthenticationError:
        if not sys.stdin.isatty():
            raise SystemExit(
                "Garmin authorization is required, but input is not interactive. "
                "Run from a terminal or provide a local FIT file."
            )

    print("Garmin Connect authorization required.", file=sys.stderr)
    email = prompt_line("Garmin email: ")
    password = getpass.getpass("Garmin password: ")
    if not email or not password:
        raise SystemExit("Garmin email and password are required")
    client = Garmin(email, password, prompt_mfa=prompt_mfa)
    client.login(str(token_store))
    return client


def video_window(metadata: dict, default_timezone: str) -> tuple[datetime, datetime]:
    # Import the production parser so download selection and synchronization
    # interpret QuickTime timestamps identically.
    from video_speed_fit import parse_tz, video_window as parse_video_window

    return parse_video_window(metadata, parse_tz(default_timezone))


def download_matching_fit(
    metadata: dict,
    default_timezone: str,
    token_store: Path,
    cache_dir: Path,
    activity_id: str | None = None,
    max_gap: float = 300.0,
) -> Path:
    start, end = video_window(metadata, default_timezone)
    cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(cache_dir, 0o700)

    if activity_id is not None:
        activity_id = str(activity_id)
        output = cache_dir / f"{activity_id}.fit"
        if output.exists():
            print(f"Using cached Garmin FIT: {output}", file=sys.stderr)
            return output

    client = connect(token_store)

    if activity_id is None:
        search_start = (start - timedelta(days=1)).date().isoformat()
        search_end = (end + timedelta(days=1)).date().isoformat()
        activities = client.get_activities_by_date(search_start, search_end)
        selected = choose_activity(activities, start, end, max_gap=max_gap)
        activity_id = selected.activity_id
        overlap = overlap_seconds(start, end, selected.start, selected.end)
        print(
            f"Selected Garmin activity {selected.activity_id} ({selected.name}): "
            f"{selected.start.isoformat()} to {selected.end.isoformat()}, "
            f"{overlap:.1f}s video overlap",
            file=sys.stderr,
        )
    else:
        activity_id = str(activity_id)
        print(f"Using Garmin activity {activity_id}", file=sys.stderr)

    output = cache_dir / f"{activity_id}.fit"
    if output.exists():
        print(f"Using cached Garmin FIT: {output}", file=sys.stderr)
        return output

    payload = client.download_activity(
        activity_id, dl_fmt=client.ActivityDownloadFormat.ORIGINAL
    )
    fit_data = extract_fit(payload)
    output.write_bytes(fit_data)
    os.chmod(output, 0o600)
    print(f"Downloaded Garmin FIT: {output}", file=sys.stderr)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--default-timezone", default="UTC")
    parser.add_argument("--activity-id")
    parser.add_argument("--token-store", default="~/.garminconnect")
    parser.add_argument("--cache-dir", default="~/.cache/refit/garmin")
    parser.add_argument("--max-gap", type=float, default=300.0)
    args = parser.parse_args()
    path = download_matching_fit(
        json.loads(args.metadata_json)[0],
        args.default_timezone,
        Path(args.token_store).expanduser(),
        Path(args.cache_dir).expanduser(),
        args.activity_id,
        args.max_gap,
    )
    print(path)


if __name__ == "__main__":
    main()
