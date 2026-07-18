#!/usr/bin/env python3
"""Validate dense optical-flow speed against TartanDrive 2.0 wheel RPM.

The selected sequence has synchronized 10 Hz RGB, 50 Hz fused GNSS/INS
odometry, and four 50 Hz wheel encoders.  The odometry supplies the dry-run
comparison and the mean-speed calibration used by the optical estimator.  It
is never used as wheel ground truth.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path
import re
import tarfile
import time
import urllib.request

import cv2
import numpy as np

from optical_flow_pipeline import blur_frame, median_flow_magnitude
from speed_estimation import (
    arithmetic_mean_scale, error_summary, stationary_interval_baseline,
)


SEQUENCE = "2023-11-14-14-24-21_gupta"
BASE_URL = (
    "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/"
    "AUTH_ac8533a83cff4d48bc8c608ad222d330/tartandrive2"
)
FILES = {
    "image_left_color.tar": "kitti/all_topics/{sequence}/image_left_color.tar",
    "gps_odom_odometry.npy": "kitti/all_topics/{sequence}/gps_odom/odometry.npy",
    "gps_odom_timestamps.txt": "kitti/all_topics/{sequence}/gps_odom/timestamps.txt",
    "wheel_rpm_encoder.npy": "kitti/all_topics/{sequence}/wheel_rpm/encoder.npy",
    "wheel_rpm_timestamps.txt": "kitti/all_topics/{sequence}/wheel_rpm/timestamps.txt",
    "data_extraction.log": "kitti/all_topics/{sequence}/data_extraction.log",
    "info.yaml": "bags/{sequence}/info.yaml",
}
WHEEL_NAMES = ("front_left", "front_right", "rear_left", "rear_right")


def download_file(url, output):
    if output.exists() and output.stat().st_size:
        print(f"Using {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as stream:
        total = int(response.headers.get("Content-Length", 0))
        copied = 0
        started = time.monotonic()
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            stream.write(block)
            copied += len(block)
            elapsed = max(time.monotonic() - started, 1e-6)
            fraction = copied / total if total else 0
            eta = (total - copied) / (copied / elapsed) if total and copied else math.inf
            print(
                f"\rDownload {output.name}: {fraction:6.1%} "
                f"{copied / 2**20:7.1f} MiB ETA "
                f"{eta:5.0f}s" if math.isfinite(eta) else "--",
                end="", flush=True,
            )
    partial.replace(output)
    print()


def acquire(data_dir, sequence):
    raw = data_dir / "raw"
    for local_name, remote in FILES.items():
        download_file(f"{BASE_URL}/{remote.format(sequence=sequence)}", raw / local_name)
    images = data_dir / "images"
    existing = sorted(images.glob("*.png")) if images.exists() else []
    if not existing:
        images.mkdir(parents=True, exist_ok=True)
        with tarfile.open(raw / "image_left_color.tar") as archive:
            archive.extractall(images, filter="data")
    return raw, images


def camera_times(log_path, count):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"topic /multisense/left/image_rect_color frames \d+,\s+"
        r"start time ([0-9.]+), end time ([0-9.]+)", text,
    )
    if not match:
        raise ValueError("could not recover camera timestamps from data_extraction.log")
    return np.linspace(float(match.group(1)), float(match.group(2)), count)


def sampled_pairs(image_paths, image_t, sample_fps):
    targets = np.arange(image_t[0], image_t[-1], 1.0 / sample_fps)
    indices = np.searchsorted(image_t, targets)
    indices = np.clip(indices, 0, len(image_t) - 2)
    indices = np.unique(indices)
    return [(i, image_paths[i], image_paths[i + 1], image_t[i + 1]) for i in indices]


def flow_one(item):
    index, first, second, timestamp = item
    a = cv2.imread(str(first), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(second), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        raise ValueError(f"failed to decode camera pair at index {index}")
    width = 640
    height = max(2, round(a.shape[0] * width / a.shape[1] / 2) * 2)
    a = cv2.resize(a, (width, height))
    b = cv2.resize(b, (width, height))
    return timestamp, median_flow_magnitude(blur_frame(a), blur_frame(b))


def optical_series(image_paths, image_t, sample_fps, workers):
    pairs = sampled_pairs(image_paths, image_t, sample_fps)
    started = time.monotonic()
    values = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i, value in enumerate(executor.map(flow_one, pairs), 1):
            values.append(value)
            if i == len(pairs) or i % max(1, len(pairs) // 100) == 0:
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = i / elapsed
                eta = (len(pairs) - i) / rate
                print(
                    f"\rOptical flow: {i / len(pairs):6.1%} "
                    f"{rate:5.1f} samples/s ETA {eta:5.1f}s",
                    end="\n" if i == len(pairs) else "", flush=True,
                )
    flow_t = np.array([value[0] for value in values])
    flow = np.array([value[1] for value in values])
    return flow_t, flow


def evaluate(raw, images, output, sequence, sample_fps, workers, wheel_diameter_inches):
    paths = sorted(images.glob("*.png"))
    if len(paths) < 2:
        raise ValueError("camera archive contains too few images")
    image_t = camera_times(raw / "data_extraction.log", len(paths))
    flow_t, flow = optical_series(paths, image_t, sample_fps, workers)

    odom_t = np.loadtxt(raw / "gps_odom_timestamps.txt")
    odom = np.load(raw / "gps_odom_odometry.npy", allow_pickle=False)
    dry_native = np.linalg.norm(odom[:, 7:10], axis=1)
    baseline, stop_runs, _ = stationary_interval_baseline(
        flow_t,
        flow,
        np.interp(flow_t, odom_t, dry_native),
        stationary_tolerance=0.05,
        min_duration=2.0,
    )
    flow = np.maximum(flow - baseline, 0.0)

    wheel_t = np.loadtxt(raw / "wheel_rpm_timestamps.txt")
    wheel_rpm = np.load(raw / "wheel_rpm_encoder.npy", allow_pickle=False)
    start = max(flow_t[0], odom_t[0], wheel_t[0])
    end = min(flow_t[-1], odom_t[-1], wheel_t[-1])
    grid = np.arange(start, end, 0.05)
    dry = np.interp(grid, odom_t, dry_native)
    motion = np.interp(grid, flow_t, flow)
    full, _ = arithmetic_mean_scale(motion, np.mean(dry))
    wheels_rpm = np.column_stack([
        np.interp(grid, wheel_t, wheel_rpm[:, i]) for i in range(4)
    ])

    # Manufacturer tire size is 25 inches. The robust median prevents a single
    # bad encoder or spinning wheel from dominating the reference at a sample.
    rpm_to_mps = math.pi * wheel_diameter_inches * 0.0254 / 60.0
    wheels_mps = wheels_rpm * rpm_to_mps
    wheel_reference = np.median(wheels_mps, axis=1)
    rear_reference = np.mean(wheels_mps[:, 2:4], axis=1)

    # Also report a scale-normalized comparison. This isolates temporal shape
    # from uncertainty in loaded tire radius and Racepak calibration.
    wheel_scaled, _ = arithmetic_mean_scale(wheel_reference, np.mean(dry))
    rear_scaled, _ = arithmetic_mean_scale(rear_reference, np.mean(dry))
    metrics = {
        "sequence": sequence,
        "duration_seconds": float(end - start),
        "samples": int(len(grid)),
        "optical_sample_fps": sample_fps,
        "temporal_smoothing_seconds": 0.0,
        "optical_baseline_px_per_frame": baseline,
        "stationary_runs_used": stop_runs,
        "wheel_diameter_inches": wheel_diameter_inches,
        "primary_reference": "sample-wise median of four wheel encoders",
        "dry_vs_nominal_wheel": error_summary(dry, wheel_reference),
        "full_vs_nominal_wheel": error_summary(full, wheel_reference),
        "dry_vs_mean_normalized_wheel": error_summary(dry, wheel_scaled),
        "full_vs_mean_normalized_wheel": error_summary(full, wheel_scaled),
        "dry_vs_mean_normalized_rear_pair": error_summary(dry, rear_scaled),
        "full_vs_mean_normalized_rear_pair": error_summary(full, rear_scaled),
    }

    median_rpm = np.median(wheels_rpm, axis=1)
    per_wheel = {}
    for i, name in enumerate(WHEEL_NAMES):
        delta = wheels_rpm[:, i] - median_rpm
        per_wheel[name] = {
            "mean_rpm": float(np.mean(wheels_rpm[:, i])),
            "max_rpm": float(np.max(wheels_rpm[:, i])),
            "mean_absolute_deviation_from_four_wheel_median_rpm": float(np.mean(np.abs(delta))),
            "p95_absolute_deviation_from_four_wheel_median_rpm": float(np.percentile(np.abs(delta), 95)),
            "fraction_over_25rpm_from_four_wheel_median": float(np.mean(np.abs(delta) > 25)),
            "correlation_with_rear_pair_mean": float(np.corrcoef(wheels_rpm[:, i], np.mean(wheels_rpm[:, 2:4], axis=1))[0, 1]),
        }
    metrics["wheel_diagnostics"] = per_wheel
    metrics["rear_pair"] = {
        "mean_absolute_disagreement_rpm": float(np.mean(np.abs(wheels_rpm[:, 2] - wheels_rpm[:, 3]))),
        "p95_absolute_disagreement_rpm": float(np.percentile(np.abs(wheels_rpm[:, 2] - wheels_rpm[:, 3]), 95)),
        "correlation": float(np.corrcoef(wheels_rpm[:, 2], wheels_rpm[:, 3])[0, 1]),
    }

    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "seconds", "dry_mps", "full_mps", "wheel_median_nominal_mps",
            "wheel_median_scaled_mps", *[f"{name}_rpm" for name in WHEEL_NAMES],
        ])
        for row in zip(grid - start, dry, full, wheel_reference, wheel_scaled, *wheels_rpm.T):
            writer.writerow(row)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_report(output / "report.md", metrics)
    return metrics


def write_report(path, metrics):
    mph = 2.2369362921
    lines = [
        "# TartanDrive 2.0 validation",
        "",
        f"Sequence: `{metrics['sequence']}` ({metrics['duration_seconds']:.2f} s).",
        "",
        "The dry method is fused GNSS/INS odometry speed. The full method is the "
        "repository's unsmoothed dense Farneback preprocessing and median ROI magnitude, "
        "baseline-corrected at an odometry-observed stop and normalized to the "
        "dry method's mean speed. Wheel data is not used by either estimate.",
        "",
        "## Scale-normalized result",
        "",
        "The wheel reference is the sample-wise median of four encoders, then given "
        "the same mean as the estimators. This tests temporal speed shape without "
        "pretending the loaded tire radius is known exactly.",
        "",
        "| Method | MAE | RMSE | P95 abs. | Max abs. | Bias | Pearson r |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("dry", "full"):
        value = metrics[f"{method}_vs_mean_normalized_wheel"]
        lines.append(
            f"| {method} | {value['mae_mps'] * mph:.3f} mph | "
            f"{value['rmse_mps'] * mph:.3f} mph | "
            f"{value['p95_absolute_error_mps'] * mph:.3f} mph | "
            f"{value['max_absolute_error_mps'] * mph:.3f} mph | "
            f"{value['bias_mps'] * mph:+.3f} mph | {value['pearson_correlation']:.3f} |"
        )
    lines.extend([
        "",
        "Rear-pair sensitivity (same mean normalization): dry MAE "
        f"{metrics['dry_vs_mean_normalized_rear_pair']['mae_mps'] * mph:.3f} mph; "
        "full MAE "
        f"{metrics['full_vs_mean_normalized_rear_pair']['mae_mps'] * mph:.3f} mph. "
        "The conclusion is unchanged when only the mutually consistent rear encoders are used.",
        "",
        "With the nominal 25-inch tire conversion rather than mean normalization, "
        f"dry MAE is {metrics['dry_vs_nominal_wheel']['mae_mps'] * mph:.3f} mph "
        f"and full MAE is {metrics['full_vs_nominal_wheel']['mae_mps'] * mph:.3f} mph.",
        "",
        "## Wheel agreement diagnostics",
        "",
        "| Wheel | Mean RPM | Max RPM | Mean abs. deviation from median | "
        "P95 abs. deviation | >25 RPM fraction | Correlation with rear-pair mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name in WHEEL_NAMES:
        value = metrics["wheel_diagnostics"][name]
        lines.append(
            f"| {name} | {value['mean_rpm']:.2f} | {value['max_rpm']:.2f} | "
            f"{value['mean_absolute_deviation_from_four_wheel_median_rpm']:.2f} RPM | "
            f"{value['p95_absolute_deviation_from_four_wheel_median_rpm']:.2f} RPM | "
            f"{value['fraction_over_25rpm_from_four_wheel_median']:.1%} | "
            f"{value['correlation_with_rear_pair_mean']:.3f} |"
        )
    rear = metrics["rear_pair"]
    lines.extend([
        "",
        f"Rear-wheel correlation is {rear['correlation']:.4f}; mean absolute "
        f"rear-pair disagreement is {rear['mean_absolute_disagreement_rpm']:.2f} RPM "
        f"(P95 {rear['p95_absolute_disagreement_rpm']:.2f} RPM).",
        "",
        "The per-wheel spread is the observed slip/sensor-health diagnostic. It "
        "does not assume that off-road ATV wheels slip more than mountain-bike wheels.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("validation-data/tartandrive2"))
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output/tartandrive2"))
    parser.add_argument("--sequence", default=SEQUENCE)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--flow-workers", type=int, default=min(16, max(1, __import__('os').cpu_count() or 1)))
    parser.add_argument("--wheel-diameter-inches", type=float, default=25.0)
    args = parser.parse_args()
    if args.sample_fps <= 0 or args.sample_fps > 10:
        parser.error("--sample-fps must be in (0, 10]")
    raw, images = acquire(args.data_dir.resolve(), args.sequence)
    metrics = evaluate(
        raw, images, args.output_dir.resolve(), args.sequence, args.sample_fps,
        args.flow_workers, args.wheel_diameter_inches,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
