#!/usr/bin/env python3
"""Create a time-clipped GPX whose speed follows video-derived optical motion."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import subprocess
import sys
import time
from bisect import bisect_left
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit("Install dependencies with: python3 -m pip install opencv-python numpy") from exc

from optical_flow_pipeline import blur_frame, median_flow_magnitude

GPXTPX = "http://www.garmin.com/xmlschemas/TrackPointExtension/v1"
ET.register_namespace("gpxtpx", GPXTPX)


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def namespace(tag: str) -> str:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") else ""


def qualified(parent, name: str) -> str:
    uri = namespace(parent.tag)
    return f"{{{uri}}}{name}" if uri else name


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


def parse_gpx_time(text: str) -> datetime:
    return parse_datetime(text, timezone.utc)


def child(point, name):
    return next((x for x in point if local(x.tag) == name), None)


def point_time(point) -> datetime:
    node = child(point, "time")
    if node is None or not node.text:
        raise ValueError("Every GPX track point must contain <time>")
    return parse_gpx_time(node.text)


def haversine(a, b) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371008.8 * 2 * math.asin(math.sqrt(h))


def interpolate_point(a, b, when: datetime):
    ta, tb = point_time(a), point_time(b)
    f = (when - ta).total_seconds() / (tb - ta).total_seconds()
    out = copy.deepcopy(a if f < 0.5 else b)
    out.set("lat", f"{float(a.get('lat')) + f * (float(b.get('lat')) - float(a.get('lat'))):.9f}")
    out.set("lon", f"{float(a.get('lon')) + f * (float(b.get('lon')) - float(a.get('lon'))):.9f}")
    ea, eb = child(a, "ele"), child(b, "ele")
    eo = child(out, "ele")
    if ea is not None and eb is not None and eo is not None and ea.text and eb.text:
        eo.text = f"{float(ea.text) + f * (float(eb.text) - float(ea.text)):.3f}"
    child(out, "time").text = when.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return out


def boundary_point(points, times, when):
    i = bisect_left(times, when)
    if i < len(times) and times[i] == when:
        return copy.deepcopy(points[i])
    if i == 0 or i == len(times):
        raise ValueError("Video window is not fully contained in GPX time range")
    return interpolate_point(points[i - 1], points[i], when)


def intersect_window(video_start, video_end, gpx_start, gpx_end):
    start = max(video_start, gpx_start)
    end = min(video_end, gpx_end)
    if start >= end:
        raise ValueError(
            "Video and GPX time ranges do not overlap: "
            f"video {video_start.isoformat()} to {video_end.isoformat()}, "
            f"GPX {gpx_start.isoformat()} to {gpx_end.isoformat()}"
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
    floor = float(np.percentile(vals, 5))
    vals = np.maximum(vals - floor, 0.0)
    return np.array([x[0] for x in samples]), vals


def speed_series(points, times):
    """Return GPX epoch seconds and reported speed, or GPS-derived speed."""
    epoch = np.array([t.timestamp() for t in times])
    reported = []
    for point in points:
        node = next((x for x in point.iter() if local(x.tag).lower() == "speed"), None)
        try:
            reported.append(float(node.text) if node is not None and node.text else math.nan)
        except ValueError:
            reported.append(math.nan)
    reported = np.array(reported)
    valid = np.isfinite(reported)
    if valid.sum() >= 2:
        values = np.interp(epoch, epoch[valid], reported[valid])
        source = "reported GPX speed"
    else:
        dt = np.diff(epoch)
        distances = np.array([
            haversine((float(a.get("lat")), float(a.get("lon"))),
                      (float(b.get("lat")), float(b.get("lon"))))
            for a, b in zip(points, points[1:])
        ])
        valid_intervals = dt > 0
        if valid_intervals.sum() < 2:
            raise ValueError("GPX has too few positive-duration intervals for clock alignment")
        epoch = ((epoch[:-1] + epoch[1:]) / 2)[valid_intervals]
        values = (distances[valid_intervals] / dt[valid_intervals])
        source = "GPS-derived speed"

    # Resample and smooth GPS jitter before comparing its shape with optical motion.
    uniform_t = np.arange(math.ceil(epoch[0]), math.floor(epoch[-1]) + 1, dtype=float)
    uniform_v = np.interp(uniform_t, epoch, values)
    if len(uniform_v) >= 7:
        uniform_v = np.convolve(uniform_v, np.ones(7) / 7, mode="same")
    return uniform_t, uniform_v, source


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


def replace_speed(point, speed: float):
    matches = [x for x in point.iter() if local(x.tag).lower() == "speed"]
    if matches:
        for node in matches:
            node.text = f"{speed:.3f}"
        return
    ext = child(point, "extensions")
    if ext is None:
        ext = ET.SubElement(point, qualified(point, "extensions"))
    tpx = next((x for x in ext if local(x.tag) == "TrackPointExtension"), None)
    if tpx is None:
        tpx = ET.SubElement(ext, f"{{{GPXTPX}}}TrackPointExtension")
    ET.SubElement(tpx, f"{{{GPXTPX}}}speed").text = f"{speed:.3f}"


def write_compatible_gpx(tree, input_path: str, output_path: str):
    """Serialize with the namespace prefixes/declarations used by the input GPX."""
    document_namespace = namespace(tree.getroot().tag)
    if document_namespace:
        ET.register_namespace("", document_namespace)
    source_head = Path(input_path).read_text(encoding="utf-8")[:8192]
    declarations = re.findall(r'xmlns:([A-Za-z_][\w.-]*)=["\']([^"\']+)["\']', source_head)
    tpx_prefix = next((prefix for prefix, uri in declarations if uri == GPXTPX), "gpxtpx")

    import io
    buffer = io.BytesIO()
    tree.write(buffer, encoding="utf-8", xml_declaration=True)
    text = buffer.getvalue().decode("utf-8")
    if tpx_prefix != "gpxtpx":
        text = text.replace("xmlns:gpxtpx=", f"xmlns:{tpx_prefix}=")
        text = text.replace("gpxtpx:", f"{tpx_prefix}:")
    for prefix, uri in declarations:
        if f"xmlns:{prefix}=" not in text:
            text = text.replace("<gpx ", f'<gpx xmlns:{prefix}="{uri}" ', 1)
    Path(output_path).write_text(text, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--gpx", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metadata-json", required=True)
    p.add_argument("--default-timezone", default="UTC")
    p.add_argument("--sample-fps", type=float, default=4.0)
    p.add_argument(
        "--dry-run", action="store_true",
        help="auto-sync and write clipped original GPX without replacing speeds",
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
    tree = ET.parse(args.gpx)
    root = tree.getroot()
    # Keep GPX as the document's default namespace. Although an ns0-prefixed
    # document is equivalent XML, Insta360 Studio's importer is prefix-sensitive.
    root_namespace = namespace(root.tag)
    if root_namespace:
        ET.register_namespace("", root_namespace)
    track_points = [x for x in root.iter() if local(x.tag) == "trkpt"]
    if len(track_points) < 2:
        raise ValueError("GPX must contain at least two track points")
    times = [point_time(x) for x in track_points]
    if times != sorted(times):
        raise ValueError("GPX track point times must be ordered")

    analysis_fps = min(args.sample_fps, 1.0) if args.dry_run else args.sample_fps
    motion_t, motion_v = optical_motion(
        args.video, analysis_fps, args.flow_workers,
        parallel_decode=args.dry_run,
    )
    if args.sync_range:
        gps_t, gps_v, speed_source = speed_series(track_points, times)
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

    start, end = intersect_window(video_start, video_end, times[0], times[-1])
    if start != video_start or end != video_end:
        print(
            "Warning: using only the video/GPX overlap: "
            f"{start.isoformat()} to {end.isoformat()}",
            file=sys.stderr,
        )

    selected = [boundary_point(track_points, times, start)]
    selected.extend(copy.deepcopy(p) for p, t in zip(track_points, times) if start < t < end)
    selected.append(boundary_point(track_points, times, end))
    selected_times = [point_time(x) for x in selected]
    coords = [(float(x.get("lat")), float(x.get("lon"))) for x in selected]
    distance = sum(haversine(a, b) for a, b in zip(coords, coords[1:]))
    duration = (end - start).total_seconds()
    avg_speed = distance / duration

    # Optical-motion times are relative to the original video, even when the
    # beginning or end of the output is clipped to the GPX range.
    offsets = np.array([(t - video_start).total_seconds() for t in selected_times])
    if args.dry_run:
        print("Dry-run mode: preserving original GPX speeds and positions")
    else:
        relative = np.interp(offsets, motion_t, motion_v, left=motion_v[0], right=motion_v[-1])
        dt = np.diff(offsets)
        interval_motion = (relative[:-1] + relative[1:]) / 2
        weighted_mean = float(np.sum(interval_motion * dt) / np.sum(dt))
        if weighted_mean <= 1e-12:
            print("Warning: no usable optical motion; writing constant average speed", file=sys.stderr)
            speeds = np.full(len(selected), avg_speed)
        else:
            speeds = relative * (avg_speed / weighted_mean)
        for point, speed in zip(selected, speeds):
            replace_speed(point, float(speed))

    # Preserve document-level metadata and track structure; replace trkpt content in
    # the first segment and remove other segments so output is exactly one video window.
    segments = [x for x in root.iter() if local(x.tag) == "trkseg"]
    if not segments:
        raise ValueError("GPX contains no <trkseg>")
    for segment in segments:
        for node in list(segment):
            if local(node.tag) == "trkpt":
                segment.remove(node)
    segments[0].extend(selected)
    for parent in root.iter():
        for segment in list(parent):
            if local(segment.tag) == "trkseg" and segment is not segments[0]:
                parent.remove(segment)

    # Insta360 aligns GPX timestamps against the uncorrected MP4 clock. Encode
    # the detected video-clock correction by shifting track times oppositely.
    if clock_offset:
        timestamp_shift = timedelta(seconds=-clock_offset)
        for point in selected:
            time_node = child(point, "time")
            shifted = point_time(point) + timestamp_shift
            time_node.text = shifted.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        print(f"Shifted output GPX timestamps by {-clock_offset:+.2f}s for the MP4 clock")

    ET.indent(tree, space="  ")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_compatible_gpx(tree, args.gpx, args.output)
    print(f"Video UTC window: {video_start.isoformat()} to {video_end.isoformat()}")
    print(f"Output UTC overlap: {start.isoformat()} to {end.isoformat()}")
    print(f"GPX segment: {distance:.1f} m over {duration:.1f} s; mean {avg_speed:.3f} m/s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
