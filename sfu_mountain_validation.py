#!/usr/bin/env python3
"""Validate GPS and dense optical-flow speed against SFU wheel odometry.

The adapter operates directly on the SFU Mountain Dataset's timestamp-named
JPEGs and header-labelled CSV exports.  It intentionally does not require ROS.
"""

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

import cv2
import numpy as np

from optical_flow_pipeline import blur_frame, median_flow_magnitude, resize_frame, grayscale_frame


TORRENT_HASH = "e3d6b8d9e87cab68c7947e800e337e58fc8d8e59"
TORRENT_URL = f"https://academictorrents.com/download/{TORRENT_HASH}.torrent"
SESSION = "dry-b"
ARCHIVES = {
    "camera_3hz": "camera_stereo_left-dry-b-10th.tar",
    "camera_30hz": "camera_stereo_left-dry-b.tar",
    "gps": "navsat_fix-dry-b.tgz",
    "gps_enu": "navsat_enu-dry-b.tgz",
    "wheel": "encoder-dry-b.tgz",
}


def normalized(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def timestamp_seconds(value):
    value = float(value)
    # SFU CSV and JPEG timestamps are Unix nanoseconds.  Retain tolerance for
    # exports that used seconds or microseconds instead.
    if value > 1e17:
        return value / 1e9
    if value > 1e14:
        return value / 1e6
    if value > 1e11:
        return value / 1e3
    return value


def member_timestamp(name):
    stem = Path(name).stem
    numbers = re.findall(r"\d+(?:\.\d+)?", stem)
    if not numbers:
        return None
    return timestamp_seconds(max(numbers, key=len))


def torrent_file_indices(torrent):
    result = subprocess.run(
        ["aria2c", "--show-files", str(torrent)],
        check=True, capture_output=True, text=True,
    )
    found = {}
    for line in result.stdout.splitlines():
        match = re.match(r"\s*(\d+)\|.*?/([^/]+)$", line)
        if match:
            found[match.group(2)] = int(match.group(1))
    return found


def download_subset(data_dir, camera_rate):
    """Selectively download the public torrent without fetching its 522 GB."""
    if shutil.which("aria2c") is None:
        raise RuntimeError("aria2c is required for --download (for macOS: brew install aria2)")
    data_dir.mkdir(parents=True, exist_ok=True)
    torrent = data_dir / "sfu-mountain.torrent"
    if not torrent.exists():
        print(f"Downloading torrent metadata: {TORRENT_URL}")
        urllib.request.urlretrieve(TORRENT_URL, torrent)
    indices = torrent_file_indices(torrent)
    wanted = [ARCHIVES[f"camera_{camera_rate}hz"], ARCHIVES["gps"], ARCHIVES["wheel"]]
    missing = [name for name in wanted if name not in indices]
    if missing:
        raise RuntimeError(f"torrent metadata does not contain: {', '.join(missing)}")
    selection = ",".join(str(indices[name]) for name in wanted)
    subprocess.run([
        "aria2c", "--seed-time=0", f"--select-file={selection}",
        f"--dir={data_dir}", str(torrent),
    ], check=True)
    return data_dir / "sfu-mountain-torrent"


def locate(root, name):
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"missing {name} below {root}; run with --download")
    return matches[0]


def read_csv_archive(path):
    with tarfile.open(path) as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one CSV in {path}, found {len(members)}")
        fileobj = archive.extractfile(members[0])
        text = io.TextIOWrapper(fileobj, encoding="utf-8-sig", newline="")
        rows = list(csv.DictReader(text))
    if not rows:
        raise ValueError(f"empty sensor CSV in {path}")
    return rows


def pick_column(rows, exact=(), contains=()):
    columns = {normalized(name): name for name in rows[0]}
    for candidate in exact:
        if normalized(candidate) in columns:
            return columns[normalized(candidate)]
    for fragments in contains:
        fragments = tuple(normalized(x) for x in fragments)
        for key, original in columns.items():
            if all(fragment in key for fragment in fragments):
                return original
    raise ValueError(f"could not identify column among: {', '.join(rows[0])}")


def load_gps(path):
    rows = read_csv_archive(path)
    time_col = pick_column(rows, ("timestamp", "time"), (("stamp",), ("time",)))
    lat_col = pick_column(rows, ("latitude",), (("latitude",), ("lat",)))
    lon_col = pick_column(rows, ("longitude",), (("longitude",), ("lon",)))
    values = np.array([
        (timestamp_seconds(row[time_col]), float(row[lat_col]), float(row[lon_col]))
        for row in rows
    ], dtype=float)
    return values[np.argsort(values[:, 0])]


def load_wheel(path):
    rows = read_csv_archive(path)
    time_col = pick_column(rows, ("timestamp", "time"), (("stamp",), ("time",)))
    speed_col = pick_column(
        rows,
        ("x velocity (m/s)", "linear_x", "x_velocity"),
        (("linear", "x"), ("velocity", "x"), ("linear",), ("speed",)),
    )
    values = np.array([
        (timestamp_seconds(row[time_col]), abs(float(row[speed_col]))) for row in rows
    ], dtype=float)
    return values[np.argsort(values[:, 0])]


def haversine_series(latitude, longitude):
    lat = np.radians(latitude)
    lon = np.radians(longitude)
    dlat, dlon = np.diff(lat), np.diff(lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    return 6371008.8 * 2 * np.arcsin(np.sqrt(a))


def gps_speed(gps):
    dt = np.diff(gps[:, 0])
    distance = haversine_series(gps[:, 1], gps[:, 2])
    valid = dt > 0
    times = (gps[:-1, 0] + gps[1:, 0]) / 2
    return times[valid], distance[valid] / dt[valid]


def camera_members(path):
    with tarfile.open(path) as archive:
        members = []
        for member in archive.getmembers():
            if member.isfile() and member.name.lower().endswith((".jpg", ".jpeg")):
                when = member_timestamp(member.name)
                if when is not None:
                    members.append((when, member.name))
    members.sort()
    if len(members) < 2:
        raise ValueError(f"no timestamp-named JPEG sequence found in {path}")
    return members


def choose_window(camera, gps_t, gps_v, wheel, duration):
    start = max(camera[0][0], gps_t[0], wheel[0, 0])
    end = min(camera[-1][0], gps_t[-1], wheel[-1, 0])
    if end - start < duration:
        return start, end
    candidates = np.arange(math.ceil(start), math.floor(end - duration) + 1, 5.0)
    # Select without consulting wheel truth: favor a moving interval with speed
    # variation so clock alignment and scale are both identifiable.
    scores = []
    for candidate in candidates:
        grid = np.arange(candidate, candidate + duration, 1.0)
        speed = np.interp(grid, gps_t, gps_v)
        scores.append(np.std(speed) + 0.15 * np.mean(speed))
    best = float(candidates[int(np.argmax(scores))])
    return best, best + duration


def optical_series(archive_path, members, start, end):
    selected = [(t, name) for t, name in members if start <= t <= end]
    times, values = [], []
    previous = None
    with tarfile.open(archive_path) as archive:
        for index, (when, name) in enumerate(selected, 1):
            data = np.frombuffer(archive.extractfile(name).read(), dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"cannot decode {name}")
            gray = blur_frame(grayscale_frame(resize_frame(frame)))
            if previous is not None:
                values.append(median_flow_magnitude(previous, gray))
                times.append(when)
            previous = gray
            if index % 100 == 0 or index == len(selected):
                print(f"\rSFU optical flow: {index}/{len(selected)} frames", end="", flush=True)
    print()
    times, values = np.asarray(times), np.asarray(values)
    if len(values) < 2:
        raise ValueError("selected SFU interval has too few images")
    median_dt = float(np.median(np.diff(times)))
    kernel = max(3, round(2.0 / median_dt) | 1)
    return times, np.convolve(values, np.ones(kernel) / kernel, mode="same")


def rank_correlation(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return -1.0
    ar = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort")
    br = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort")
    return float(np.corrcoef(ar, br)[0, 1])


def clock_offset(opt_t, optical, gps_t, gps_v, search_range):
    relative = opt_t - opt_t[0]
    grid = np.arange(math.ceil(relative[0]), math.floor(relative[-1]) + 1)
    motion = np.interp(grid, relative, optical)
    # Offset means GPS time sampled for a given camera time, matching the main pipeline.
    candidates = np.arange(-search_range, search_range + 0.001, 0.25)
    scores = []
    for offset in candidates:
        query = opt_t[0] + grid + offset
        valid = (query >= gps_t[0]) & (query <= gps_t[-1])
        scores.append(rank_correlation(motion[valid], np.interp(query[valid], gps_t, gps_v)) if valid.sum() >= 20 else -2)
    coarse = float(candidates[int(np.argmax(scores))])
    fine = np.arange(coarse - 0.25, coarse + 0.2501, 0.025)
    fine_scores = []
    for offset in fine:
        query = opt_t[0] + grid + offset
        valid = (query >= gps_t[0]) & (query <= gps_t[-1])
        fine_scores.append(rank_correlation(motion[valid], np.interp(query[valid], gps_t, gps_v)) if valid.sum() >= 20 else -2)
    return float(fine[int(np.argmax(fine_scores))]), float(max(fine_scores))


def summary(estimate, truth):
    error = estimate - truth
    absolute = np.abs(error)
    return {
        "mae_mps": float(np.mean(absolute)),
        "rmse_mps": float(np.sqrt(np.mean(error ** 2))),
        "p95_absolute_error_mps": float(np.percentile(absolute, 95)),
        "max_absolute_error_mps": float(np.max(absolute)),
        "bias_mps": float(np.mean(error)),
        "pearson_correlation": float(np.corrcoef(estimate, truth)[0, 1]),
        "spearman_rank_correlation": rank_correlation(estimate, truth),
    }


def stationary_baseline(times, optical, dry_speed):
    """Match production: quietest median of real, nonzero-duration GPS stops."""
    stopped = np.isclose(dry_speed, 0.0, atol=1e-9)
    medians = []
    start = None
    for index, is_stopped in enumerate(stopped):
        if is_stopped and start is None:
            start = index
        if start is not None and (not is_stopped or index == len(stopped) - 1):
            end = index if is_stopped else index - 1
            if end > start and times[end] > times[start]:
                medians.append(float(np.median(optical[start:end + 1])))
            start = None
    return min(medians) if medians else 0.0, len(medians)


def evaluate(data_root, output_dir, camera_rate, duration, sync_range):
    camera_archive = locate(data_root, ARCHIVES[f"camera_{camera_rate}hz"])
    gps = load_gps(locate(data_root, ARCHIVES["gps"]))
    wheel = load_wheel(locate(data_root, ARCHIVES["wheel"]))
    gps_t, gps_v = gps_speed(gps)
    members = camera_members(camera_archive)
    start, end = choose_window(members, gps_t, gps_v, wheel, duration)
    opt_t, optical = optical_series(camera_archive, members, start, end)
    offset, sync_score = clock_offset(opt_t, optical, gps_t, gps_v, sync_range)

    eval_start = max(opt_t[0], wheel[0, 0], gps_t[0] - offset)
    eval_end = min(opt_t[-1], wheel[-1, 0], gps_t[-1] - offset)
    grid = np.arange(eval_start, eval_end, 0.1)
    truth = np.interp(grid, wheel[:, 0], wheel[:, 1])
    dry = np.interp(grid + offset, gps_t, gps_v)

    optical_grid = np.interp(grid, opt_t, optical)
    # Coordinate-derived GPS usually jitters even at rest. Do not invent a
    # stationary fraction or threshold: absent an actual zero-speed interval,
    # the production fallback is exactly zero.
    baseline, stop_runs = stationary_baseline(grid, optical_grid, dry)
    optical_grid = np.maximum(optical_grid - baseline, 0.0)
    gps_distance = float(np.trapezoid(dry, grid))
    optical_integral = float(np.trapezoid(optical_grid, grid))
    full = optical_grid * gps_distance / optical_integral if optical_integral > 1e-12 else np.full_like(grid, gps_distance / (grid[-1] - grid[0]))

    metrics = {
        "dataset": "SFU Mountain dry-b",
        "camera_rate_hz": camera_rate,
        "camera_frames": len(opt_t) + 1,
        "evaluation_start_unix": float(grid[0]),
        "evaluation_end_unix": float(grid[-1]),
        "clock_offset_seconds": offset,
        "clock_rank_correlation": sync_score,
        "optical_baseline_px_per_frame": baseline,
        "stationary_runs_used": stop_runs,
        "wheel_mean_mps": float(np.mean(truth)),
        "dry": summary(dry, truth),
        "full": summary(full, truth),
    }
    metrics["winner_by_mae"] = "full" if metrics["full"]["mae_mps"] < metrics["dry"]["mae_mps"] else "dry"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savetxt(
        output_dir / "comparison.csv",
        np.column_stack((grid - grid[0], truth, dry, full)),
        delimiter=",", comments="", header="seconds,wheel_mps,dry_mps,full_mps",
    )
    mph = 2.2369362921
    lines = [
        "# SFU Mountain validation", "",
        f"Session: `{SESSION}`; public camera export: {camera_rate} Hz; frames: {len(opt_t) + 1}.", "",
        "| Method | MAE | RMSE | P95 abs. | Max abs. | Bias | Pearson r | Spearman r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("dry", "full"):
        value = metrics[name]
        lines.append(
            f"| {name} | {value['mae_mps']*mph:.3f} mph | {value['rmse_mps']*mph:.3f} mph | "
            f"{value['p95_absolute_error_mps']*mph:.3f} mph | {value['max_absolute_error_mps']*mph:.3f} mph | "
            f"{value['bias_mps']*mph:+.3f} mph | {value['pearson_correlation']:.3f} | "
            f"{value['spearman_rank_correlation']:.3f} |"
        )
    lines += ["", f"Winner by MAE: **{metrics['winner_by_mae']}**.", ""]
    (output_dir / "report.md").write_text("\n".join(lines))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("validation-data/sfu-mountain"))
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output/sfu-mountain"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--camera-rate", type=int, choices=(3, 30), default=3)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--sync-range", type=float, default=30.0)
    args = parser.parse_args()
    root = download_subset(args.data_dir, args.camera_rate) if args.download else args.data_dir
    metrics = evaluate(root, args.output_dir, args.camera_rate, args.duration, args.sync_range)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
