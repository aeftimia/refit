#!/usr/bin/env python3
"""Preserve a Garmin FIT activity while aligning speed to an Insta360 video."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit("Install dependencies with: python3 -m pip install opencv-python numpy") from exc

from optical_flow_pipeline import blur_frame, median_flow_magnitude

from fit_binary import FitBinary

def parse_tz(value: str):
    if value.upper() in {"Z", "UTC", "GMT"}:
        return timezone.utc
    if re.fullmatch(r"[+-]\d\d:\d\d", value):
        sign = 1 if value[0] == "+" else -1
        return timezone(sign * timedelta(hours=int(value[1:3]), minutes=int(value[4:6])))
    return ZoneInfo(value)


def parse_datetime(value: str, default_tz) -> datetime:
    value = value.strip().replace("Z", "+00:00")
    # ExifTool commonly emits YYYY:MM:DD HH:MM:SS, optionally with fractional seconds/offset.
    value = re.sub(r"^(\d{4}):(\d{2}):(\d{2})", r"\1-\2-\3", value)
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=default_tz)
    return dt.astimezone(timezone.utc)


def video_window(metadata: dict, default_tz) -> tuple[datetime, datetime]:
    start = None
    for key in ("DateTimeOriginal", "MediaCreateDate", "TrackCreateDate", "CreateDate"):
        if metadata.get(key):
            start = parse_datetime(str(metadata[key]), default_tz)
            break
    if start is None:
        raise ValueError("No usable creation timestamp in MP4 metadata")
    duration = float(metadata.get("Duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("No positive MP4 duration in metadata")
    return start, start + timedelta(seconds=duration)


def haversine(a, b) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371008.8 * 2 * math.asin(math.sqrt(h))


def intersect_window(video_start, video_end, fit_start, fit_end):
    start = max(video_start, fit_start)
    end = min(video_end, fit_end)
    if start >= end:
        raise ValueError(
            "Video and FIT time ranges do not overlap: "
            f"video {video_start.isoformat()} to {video_end.isoformat()}, "
            f"FIT {fit_start.isoformat()} to {fit_end.isoformat()}"
        )
    return start, end


def flow_magnitude(previous, gray) -> float:
    return median_flow_magnitude(previous, gray)


def optical_motion(
    video: str, sample_fps: float, flow_workers: int,
    parallel_decode=False,
):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError(f"OpenCV cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video reports an invalid frame rate")
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = source_frames / fps
    cap.release()
    width = 640
    height = max(2, round(source_height * width / source_width / 2) * 2)
    total_frames = max(1, round(duration * sample_fps))
    sample_step = max(2, round(fps / sample_fps))
    pair_filter = (
        f"select=lt(mod(n\\,{sample_step})\\,2),"
        f"scale={width}:{height},format=gray"
    )
    samples, previous = [], None
    pending = deque()
    frame_no = 0
    progress_started = time.monotonic()
    last_progress = 0.0

    def show_progress(force=False, complete=False):
        nonlocal last_progress
        now = time.monotonic()
        if not force and now - last_progress < 0.5:
            return
        last_progress = now
        elapsed = max(now - progress_started, 1e-6)
        processed = total_frames if complete and total_frames > 0 else frame_no
        rate = processed / elapsed
        if total_frames > 0:
            fraction = min(processed / total_frames, 1.0)
            filled = round(30 * fraction)
            bar = "#" * filled + "-" * (30 - filled)
            remaining = max(total_frames - processed, 0)
            eta = remaining / rate if rate > 0 else math.inf
            eta_text = time.strftime("%M:%S", time.gmtime(eta)) if math.isfinite(eta) else "--:--"
            message = (
                f"\rOptical flow [{bar}] {fraction:6.1%} "
                f"{rate:5.1f} samples/s ETA {eta_text}"
            )
        else:
            message = f"\rOptical flow: {processed} samples, {rate:.1f} samples/s ETA unknown"
        print(message, end="\n" if complete else "", file=sys.stderr, flush=True)

    show_progress(force=True)
    frame_bytes = width * height

    if parallel_decode:
        sample_count = min(1000, max(2, int(duration * fps / 2)))
        sample_times = np.linspace(0, max(0, duration - 2 / fps), sample_count)
        total_frames = sample_count

        def decode_pair(index, sample_time):
            command = [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-ss", f"{sample_time:.6f}", "-i", video, "-frames:v", "2",
                "-vf", f"scale={width}:{height},format=gray",
                "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
            ]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode:
                error = result.stderr.decode("utf-8", errors="replace").strip()
                raise ValueError(f"FFmpeg sample {index + 1} failed: {error}")
            if len(result.stdout) < frame_bytes * 2:
                raise ValueError(f"FFmpeg sample {index + 1} returned fewer than two frames")
            frames = np.frombuffer(result.stdout[:frame_bytes * 2], dtype=np.uint8)
            return index, frames.reshape(2, height, width)

        decoded = [None] * sample_count
        decoder_count = min(flow_workers, 16, sample_count)
        with ThreadPoolExecutor(max_workers=decoder_count) as decoder_pool:
            futures = [
                decoder_pool.submit(decode_pair, i, float(sample_time))
                for i, sample_time in enumerate(sample_times)
            ]
            for future in as_completed(futures):
                index, frames = future.result()
                decoded[index] = frames
                frame_no += 1
                show_progress(force=True)

        with ThreadPoolExecutor(max_workers=flow_workers) as executor:
            for sample_time, frames in zip(sample_times, decoded):
                previous = blur_frame(frames[0])
                gray = blur_frame(frames[1])
                t = float(sample_time) + 1 / fps
                pending.append((t, executor.submit(flow_magnitude, previous, gray)))
                if len(pending) >= flow_workers * 2:
                    sample_t, future = pending.popleft()
                    samples.append((sample_t, future.result()))
            while pending:
                sample_t, future = pending.popleft()
                samples.append((sample_t, future.result()))
        show_progress(force=True, complete=True)
    else:
        command = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", video,
            "-vf", pair_filter, "-fps_mode", "vfr",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def read_frame():
            data = bytearray()
            while len(data) < frame_bytes:
                chunk = process.stdout.read(frame_bytes - len(data))
                if not chunk:
                    break
                data.extend(chunk)
            if len(data) != frame_bytes:
                return None
            return np.frombuffer(data, dtype=np.uint8).reshape(height, width)

        try:
            # OpenCV releases the GIL during Farneback, so independent sampled-frame
            # pairs can run concurrently. Keep the queue bounded to limit frame memory.
            with ThreadPoolExecutor(max_workers=flow_workers) as executor:
                while True:
                    previous = read_frame()
                    gray = read_frame()
                    if previous is None or gray is None:
                        break
                    previous = blur_frame(previous)
                    gray = blur_frame(gray)
                    t = frame_no / sample_fps + 1 / fps
                    pending.append((t, executor.submit(flow_magnitude, previous, gray)))
                    frame_no += 1
                    show_progress()
                    if len(pending) >= flow_workers * 2:
                        sample_t, future = pending.popleft()
                        samples.append((sample_t, future.result()))
                while pending:
                    sample_t, future = pending.popleft()
                    samples.append((sample_t, future.result()))
            stderr = process.communicate()[1].decode("utf-8", errors="replace").strip()
            if process.returncode:
                raise ValueError(f"FFmpeg video decoding failed: {stderr}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
        show_progress(force=True, complete=True)
    if len(samples) < 2:
        raise ValueError("Video is too short to estimate motion")
    vals = np.array([x[1] for x in samples], dtype=float)
    kernel = max(3, int(round(sample_fps * 2)) | 1)
    vals = np.convolve(vals, np.ones(kernel) / kernel, mode="same")
    # Do not assume any fixed fraction of the recording represents zero motion.
    # Until a supported baseline estimate exists, the conservative floor is zero.
    return np.array([x[0] for x in samples]), vals


def estimate_stationary_baseline(video_start, motion_t, motion_v, gps_t, raw_gps_v):
    """Use the quietest median among observed, nonzero-duration GPS stops."""
    stationary = np.isclose(raw_gps_v, 0.0, atol=1e-9)
    intervals = []
    start = None
    for i, stopped in enumerate(stationary):
        if stopped and start is None:
            start = i
        if start is not None and (not stopped or i == len(stationary) - 1):
            end = i if stopped and i == len(stationary) - 1 else i - 1
            # At least two 1 Hz timestamps establish a nonzero-duration interval.
            if end > start:
                intervals.append((gps_t[start], gps_t[end]))
            start = None

    absolute_motion_t = video_start.timestamp() + motion_t
    medians = []
    supporting_samples = 0
    for start_time, end_time in intervals:
        selected = (absolute_motion_t >= start_time) & (absolute_motion_t <= end_time)
        if selected.any():
            values = motion_v[selected]
            medians.append(float(np.median(values)))
            supporting_samples += int(selected.sum())
    if not medians:
        return 0.0, 0, 0
    return min(medians), len(medians), supporting_samples


def rank_values(values):
    """Ranks with averaged ties, sufficient for a dependency-free Spearman score."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2
        i = j
    return ranks


def correlation(a, b):
    a, b = rank_values(a), rank_values(b)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return -1.0
    return float(np.corrcoef(a, b)[0, 1])


def find_clock_offset(video_start, motion_t, motion_v, gps_t, gps_v, search_range):
    """Find the static video clock correction with maximum speed-shape correlation."""
    # One sample per second is enough for clock alignment and avoids overweighting
    # adjacent, highly autocorrelated optical-flow frames.
    seconds = np.arange(math.ceil(motion_t[0]), math.floor(motion_t[-1]) + 1, dtype=float)
    motion = np.interp(seconds, motion_t, motion_v)
    minimum = max(20, min(60, len(seconds) // 2))

    def score(offset):
        query = video_start.timestamp() + seconds + offset
        valid = (query >= gps_t[0]) & (query <= gps_t[-1])
        if valid.sum() < minimum:
            return -2.0, int(valid.sum())
        value = correlation(motion[valid], np.interp(query[valid], gps_t, gps_v))
        # Very lightly prefer candidates supported by more of the video.
        value -= 0.02 * (1 - valid.mean())
        return value, int(valid.sum())

    coarse = np.arange(-search_range, search_range + 0.25, 0.5)
    coarse_scores = [score(x)[0] for x in coarse]
    best_coarse = float(coarse[int(np.argmax(coarse_scores))])
    fine = np.arange(best_coarse - 0.5, best_coarse + 0.501, 0.05)
    candidates = [(score(float(x))[0], float(x), score(float(x))[1]) for x in fine]
    best_score, best_offset, count = max(candidates)
    zero_score, _ = score(0.0)
    return best_offset, best_score, zero_score, count, abs(best_coarse) >= search_range - 0.25


def fit_track(fit_file):
    """Return timestamped FIT records containing usable geographic positions."""
    records = fit_file.track_records()
    if len(records) < 2:
        raise ValueError("FIT must contain at least two timestamped position records")
    times = [datetime.fromtimestamp(message.timestamp / 1000, timezone.utc) for message in records]
    if times != sorted(times):
        raise ValueError("FIT position record times must be ordered")
    return records, times


def fit_speed_series(fit_file, records, times):
    """Return a uniform Garmin speed series while retaining Smart Recording input."""
    dense = fit_file.gps_metadata_points()
    if len(dense) >= 2:
        dense_t = np.array([timestamp for _, timestamp, _ in dense], dtype=float)
        dense_v = np.array([speed for _, _, speed in dense], dtype=float)
        uniform_t = np.arange(math.ceil(dense_t[0]), math.floor(dense_t[-1]) + 1, dtype=float)
        raw_uniform_v = np.interp(uniform_t, dense_t, dense_v)
        uniform_v = raw_uniform_v.copy()
        if len(uniform_v) >= 7:
            uniform_v = np.convolve(uniform_v, np.ones(7) / 7, mode="same")
        return uniform_t, uniform_v, raw_uniform_v, "Garmin FIT ~1 Hz gps_metadata speed"

    epoch = np.array([value.timestamp() for value in times], dtype=float)
    reported = np.array([
        float(message.enhanced_speed) if message.enhanced_speed is not None else math.nan
        for message in records
    ])
    valid = np.isfinite(reported)
    if valid.sum() >= 2:
        values = np.interp(epoch, epoch[valid], reported[valid])
        source = "Garmin FIT enhanced_speed"
    else:
        coordinates = [(message.position_lat, message.position_long) for message in records]
        dt = np.diff(epoch)
        distances = np.array([haversine(a, b) for a, b in zip(coordinates, coordinates[1:])])
        valid_intervals = dt > 0
        epoch = ((epoch[:-1] + epoch[1:]) / 2)[valid_intervals]
        values = distances[valid_intervals] / dt[valid_intervals]
        source = "FIT coordinate-derived speed"
    uniform_t = np.arange(math.ceil(epoch[0]), math.floor(epoch[-1]) + 1, dtype=float)
    raw_uniform_v = np.interp(uniform_t, epoch, values)
    uniform_v = raw_uniform_v.copy()
    if len(uniform_v) >= 7:
        uniform_v = np.convolve(uniform_v, np.ones(7) / 7, mode="same")
    return uniform_t, uniform_v, raw_uniform_v, source


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metadata-json", required=True)
    p.add_argument("--default-timezone", default="UTC")
    p.add_argument("--sample-fps", type=float, default=4.0)
    p.add_argument(
        "--dry-run", action="store_true",
        help="auto-sync and preserve the Garmin FIT speed instead of replacing it",
    )
    p.add_argument(
        "--sync-range", type=float, default=300.0,
        help="maximum automatic video clock correction in seconds (0 disables)",
    )
    p.add_argument(
        "--flow-workers", type=int,
        default=min(4, os.cpu_count() or 1),
        help="parallel optical-flow calculations",
    )
    args = p.parse_args()
    if args.flow_workers < 1:
        p.error("--flow-workers must be at least 1")
    if args.sync_range < 0:
        p.error("--sync-range cannot be negative")

    metadata = json.loads(args.metadata_json)[0]
    video_start, video_end = video_window(metadata, parse_tz(args.default_timezone))
    clock_offset = 0.0
    fit_file = FitBinary(args.fit)
    track_points, times = fit_track(fit_file)

    motion_t = motion_v = None
    if not (args.dry_run and args.sync_range == 0):
        analysis_fps = min(args.sample_fps, 1.0) if args.dry_run else args.sample_fps
        motion_t, motion_v = optical_motion(
            args.video, analysis_fps, args.flow_workers,
            parallel_decode=args.dry_run,
        )
    if args.sync_range:
        gps_t, gps_v, raw_gps_v, speed_source = fit_speed_series(fit_file, track_points, times)
        offset, sync_score, zero_score, sync_samples, at_limit = find_clock_offset(
            video_start, motion_t, motion_v, gps_t, gps_v, args.sync_range
        )
        clock_offset = offset
        video_start += timedelta(seconds=offset)
        video_end += timedelta(seconds=offset)
        print(
            f"Automatic clock correction: {offset:+.2f}s using {speed_source} "
            f"(rank correlation {sync_score:.3f}, uncorrected {zero_score:.3f}, "
            f"{sync_samples} samples)"
        )
        if sync_score < 0.2 or sync_score - zero_score < 0.03:
            print("Warning: automatic clock alignment has low confidence", file=sys.stderr)
        if at_limit:
            print("Warning: best clock alignment is at the search limit", file=sys.stderr)
        if not args.dry_run:
            baseline, stop_count, baseline_samples = estimate_stationary_baseline(
                video_start, motion_t, motion_v, gps_t, raw_gps_v
            )
            motion_v = np.maximum(motion_v - baseline, 0.0)
            if stop_count:
                print(
                    f"Optical baseline: {baseline:.6f} px/frame from the quietest "
                    f"of {stop_count} stationary intervals ({baseline_samples} samples)"
                )
            else:
                print("Optical baseline: 0 (no observed stationary interval)")

    start, end = intersect_window(video_start, video_end, times[0], times[-1])
    if start != video_start or end != video_end:
        print(
            "Warning: using only the video/FIT overlap: "
            f"{start.isoformat()} to {end.isoformat()}",
            file=sys.stderr,
        )

    selected = [point for point, when in zip(track_points, times) if start <= when <= end]
    selected_times = [when for when in times if start <= when <= end]
    if len(selected) < 2:
        raise ValueError("Video/FIT overlap contains too few position records")
    coords = [(point.position_lat, point.position_long) for point in selected]
    distance = sum(haversine(a, b) for a, b in zip(coords, coords[1:]))
    duration = (end - start).total_seconds()
    avg_speed = distance / duration

    # Optical-motion times are relative to the original video, even when the
    # beginning or end of the output is clipped to the FIT range.
    offsets = np.array([(t - video_start).total_seconds() for t in selected_times])
    if args.dry_run:
        print("Dry-run mode: preserving every original FIT message, speed, and position")
    else:
        relative = np.interp(offsets, motion_t, motion_v, left=motion_v[0], right=motion_v[-1])
        dt = np.diff(offsets)
        interval_motion = (relative[:-1] + relative[1:]) / 2
        weighted_mean = float(np.sum(interval_motion * dt) / np.sum(dt))
        if weighted_mean <= 1e-12:
            print("Warning: no usable optical motion; writing constant average speed", file=sys.stderr)
            scale = 0.0
            speeds = np.full(len(selected), avg_speed)
        else:
            scale = avg_speed / weighted_mean
            speeds = relative * scale
        for point, speed in zip(selected, speeds):
            fit_file.set_record_speed(point.location, float(speed))
        for message, timestamp, _ in fit_file.gps_metadata_points():
            when = datetime.fromtimestamp(timestamp, timezone.utc)
            if start <= when <= end:
                relative_time = (when - video_start).total_seconds()
                speed = (
                    avg_speed if scale == 0.0 else scale * np.interp(
                        relative_time, motion_t, motion_v,
                        left=motion_v[0], right=motion_v[-1],
                    )
                )
                fit_file.set_gps_metadata_speed(
                    message, float(speed)
                )

    # Insta360 aligns FIT records against the uncorrected MP4 clock. Shift all
    # recognized original messages while retaining Garmin-specific payloads.
    if clock_offset:
        fit_file.shift_timestamps(-clock_offset)
        print(f"Shifted output FIT timestamps by {-clock_offset:+.2f}s for the MP4 clock")

    fit_file.write(args.output)
    print(f"Video UTC window: {video_start.isoformat()} to {video_end.isoformat()}")
    print(f"Output UTC overlap: {start.isoformat()} to {end.isoformat()}")
    print(f"FIT/video overlap: {distance:.1f} m over {duration:.1f} s; mean {avg_speed:.3f} m/s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
