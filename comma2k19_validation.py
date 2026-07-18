#!/usr/bin/env python3
"""Run dry/full speed estimation against comma2k19 wheel-speed ground truth."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from fit_tool.fit_file import FitFile
from fit_tool.profile.messages.record_message import RecordMessage


GPX = "http://www.topografix.com/GPX/1/1"


def load_array(segment, relative_path):
    return np.load(Path(segment) / relative_path, allow_pickle=False)


def clock_model(segment):
    """Fit UTC seconds = slope * device boot seconds + intercept."""
    boot_time = load_array(segment, "processed_log/GNSS/live_gnss_ublox/t")
    utc_ms = load_array(segment, "processed_log/GNSS/live_gnss_ublox/value")[:, 3]
    slope, intercept = np.polyfit(boot_time, utc_ms / 1000.0, 1)
    residual = utc_ms / 1000.0 - (slope * boot_time + intercept)
    return float(slope), float(intercept), float(np.std(residual))


def prepare_video(segment, output, fps, creation_time):
    source = Path(segment) / "video.hevc"
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-r", f"{fps:.12f}", "-i", str(source), "-c:v", "copy",
        "-tag:v", "hvc1", "-metadata", f"creation_time={creation_time}",
        "-y", str(output),
    ], check=True)


def prepare_gpx(
    segment, output, epoch_slope, epoch_intercept,
    video_first_boot, video_last_boot,
):
    gnss_t = load_array(segment, "processed_log/GNSS/live_gnss_ublox/t")
    gnss = load_array(segment, "processed_log/GNSS/live_gnss_ublox/value")
    start = math.ceil(max(video_first_boot, float(gnss_t[0])))
    end = math.floor(min(video_last_boot, float(gnss_t[-1])))
    sample_t = np.arange(start, end + 1, dtype=float)
    if len(sample_t) < 2:
        raise ValueError("comma2k19 segment has insufficient video/GNSS overlap")

    latitude = np.interp(sample_t, gnss_t, gnss[:, 0])
    longitude = np.interp(sample_t, gnss_t, gnss[:, 1])
    altitude = np.interp(sample_t, gnss_t, gnss[:, 4])

    ET.register_namespace("", GPX)
    root = ET.Element(f"{{{GPX}}}gpx", {
        "version": "1.1",
        "creator": "gpx-correction comma2k19 validation",
    })
    metadata = ET.SubElement(root, f"{{{GPX}}}metadata")
    ET.SubElement(metadata, f"{{{GPX}}}name").text = "comma2k19 GNSS at 1 Hz"
    track = ET.SubElement(root, f"{{{GPX}}}trk")
    ET.SubElement(track, f"{{{GPX}}}name").text = "comma2k19 GNSS-only input"
    segment_node = ET.SubElement(track, f"{{{GPX}}}trkseg")
    for t, lat, lon, ele in zip(sample_t, latitude, longitude, altitude):
        point = ET.SubElement(segment_node, f"{{{GPX}}}trkpt", {
            "lat": f"{lat:.9f}", "lon": f"{lon:.9f}",
        })
        ET.SubElement(point, f"{{{GPX}}}ele").text = f"{ele:.3f}"
        timestamp = datetime.fromtimestamp(
            epoch_slope * t + epoch_intercept, timezone.utc
        )
        ET.SubElement(point, f"{{{GPX}}}time").text = (
            timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return sample_t


def fit_speed_series(path):
    grouped = defaultdict(list)
    for record in FitFile.from_file(str(path)).records:
        message = record.message
        if (
            isinstance(message, RecordMessage)
            and message.timestamp is not None
            and message.enhanced_speed is not None
        ):
            grouped[message.timestamp / 1000.0].append(float(message.enhanced_speed))
    if len(grouped) < 2:
        raise ValueError(f"FIT output has too few speed records: {path}")
    timestamps = np.array(sorted(grouped), dtype=float)
    speeds = np.array([np.mean(grouped[t]) for t in timestamps], dtype=float)
    return timestamps, speeds


def metric_summary(estimate, truth, relative_time):
    error = estimate - truth
    absolute = np.abs(error)
    peak = int(np.argmax(absolute))
    estimate_rank = np.argsort(np.argsort(estimate, kind="mergesort"), kind="mergesort")
    truth_rank = np.argsort(np.argsort(truth, kind="mergesort"), kind="mergesort")
    return {
        "mae_mps": float(np.mean(absolute)),
        "rmse_mps": float(np.sqrt(np.mean(error ** 2))),
        "bias_mps": float(np.mean(error)),
        "p95_absolute_error_mps": float(np.percentile(absolute, 95)),
        "max_absolute_error_mps": float(absolute[peak]),
        "max_error_at_video_seconds": float(relative_time[peak]),
        "pearson_correlation": float(np.corrcoef(estimate, truth)[0, 1]),
        "spearman_rank_correlation": float(
            np.corrcoef(estimate_rank, truth_rank)[0, 1]
        ),
    }


def evaluate(segment, dry_fit, full_fit, manifest, output_dir):
    wheel_t = load_array(segment, "processed_log/CAN/wheel_speed/t")
    wheel = np.mean(
        load_array(segment, "processed_log/CAN/wheel_speed/value"), axis=1
    )
    dry_epoch, dry_speed = fit_speed_series(dry_fit)
    full_epoch, full_speed = fit_speed_series(full_fit)

    # FIT output timestamps are shifted back onto the uncorrected MP4 clock.
    # Convert that clock to comma boot time without consulting wheel speed.
    embedded_epoch = manifest["embedded_video_start_epoch"]
    frame_boot = manifest["video_first_frame_boot_seconds"]
    dry_t = frame_boot + (dry_epoch - embedded_epoch)
    full_t = frame_boot + (full_epoch - embedded_epoch)
    start = max(wheel_t[0], dry_t[0], full_t[0])
    end = min(wheel_t[-1], dry_t[-1], full_t[-1])
    if end <= start:
        raise ValueError("FIT outputs and wheel measurements do not overlap")

    # A uniform grid makes the metrics time-weighted rather than dependent on
    # the roughly 80 Hz CAN message cadence.
    grid_t = np.arange(start, end, 0.05)
    wheel_grid = np.interp(grid_t, wheel_t, wheel)
    dry_grid = np.interp(grid_t, dry_t, dry_speed)
    full_grid = np.interp(grid_t, full_t, full_speed)
    relative_time = grid_t - frame_boot
    metrics = {
        "evaluation_start_video_seconds": float(relative_time[0]),
        "evaluation_end_video_seconds": float(relative_time[-1]),
        "evaluation_samples": int(len(grid_t)),
        "wheel_speed_mean_mps": float(np.mean(wheel_grid)),
        "dry": metric_summary(dry_grid, wheel_grid, relative_time),
        "full": metric_summary(full_grid, wheel_grid, relative_time),
    }
    metrics["winner_by_mae"] = (
        "full" if metrics["full"]["mae_mps"] < metrics["dry"]["mae_mps"] else "dry"
    )
    metrics["winner_by_max_absolute_error"] = (
        "full"
        if metrics["full"]["max_absolute_error_mps"]
        < metrics["dry"]["max_absolute_error_mps"]
        else "dry"
    )

    np.savetxt(
        output_dir / "comparison.csv",
        np.column_stack((
            relative_time, wheel_grid, dry_grid, full_grid,
            np.abs(dry_grid - wheel_grid), np.abs(full_grid - wheel_grid),
        )),
        delimiter=",",
        header=(
            "video_seconds,wheel_mps,dry_mps,full_mps,"
            "dry_absolute_error_mps,full_absolute_error_mps"
        ),
        comments="",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    mph = 2.2369362921
    report = [
        "# comma2k19 validation result",
        "",
        "Wheel speed is the mean of four CAN wheel measurements and was not used "
        "by either estimator.",
        "",
        "| Method | MAE | RMSE | P95 abs. | Max abs. | Bias | Pearson r | Spearman r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("dry", "full"):
        value = metrics[name]
        report.append(
            f"| {name} | {value['mae_mps'] * mph:.3f} mph | "
            f"{value['rmse_mps'] * mph:.3f} mph | "
            f"{value['p95_absolute_error_mps'] * mph:.3f} mph | "
            f"{value['max_absolute_error_mps'] * mph:.3f} mph | "
            f"{value['bias_mps'] * mph:+.3f} mph | "
            f"{value['pearson_correlation']:.3f} | "
            f"{value['spearman_rank_correlation']:.3f} |"
        )
    report.extend([
        "",
        f"Winner by MAE: **{metrics['winner_by_mae']}**.",
        f"Winner by maximum absolute error: "
        f"**{metrics['winner_by_max_absolute_error']}**.",
        "",
    ])
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    return metrics


def run_pipeline(script_dir, video, gpx, output, dry_run, args, sync_range):
    command = ["bash", str(script_dir / "insta360_video_speed_gpx.sh")]
    if dry_run:
        command.append("--dry-run")
    command.extend([str(video), str(gpx), str(output)])
    env = os.environ.copy()
    env.update({
        "PATH": f"{Path(sys.executable).parent}:{env.get('PATH', '')}",
        "VIDEO_TIMEZONE": "UTC",
        "SAMPLE_FPS": str(args.sample_fps),
        "FLOW_WORKERS": str(args.flow_workers),
        "SYNC_RANGE": str(sync_range),
    })
    subprocess.run(command, check=True, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("segment", type=Path, help="comma2k19 one-minute segment directory")
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output/comma2k19"))
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--flow-workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--sync-range", type=float, default=5.0)
    parser.add_argument(
        "--clock-mode", choices=("synchronized", "auto"), default="synchronized",
        help=(
            "synchronized uses comma's shared camera/GNSS clock for both methods; "
            "auto independently estimates each offset from motion correlation"
        ),
    )
    args = parser.parse_args()

    segment = args.segment.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame_t = load_array(segment, "global_pose/frame_times")
    frame_dt = float(np.median(np.diff(frame_t)))
    fps = 1.0 / frame_dt
    slope, intercept, clock_residual = clock_model(segment)
    true_epoch = slope * float(frame_t[0]) + intercept
    # QuickTime metadata in this fixture deliberately has whole-second precision,
    # leaving ordinary dry/full clock alignment to recover the fractional error.
    embedded_epoch = math.floor(true_epoch)
    creation_time = datetime.fromtimestamp(embedded_epoch, timezone.utc).isoformat()

    video = output / "video.mp4"
    gpx = output / "gnss.gpx"
    dry_fit = output / "dry.fit"
    full_fit = output / "full.fit"
    prepare_video(segment, video, fps, creation_time)
    if args.clock_mode == "synchronized":
        # comma records camera and GNSS on one boot-time clock. Anchor GPX time
        # directly to the MP4 clock so both estimators see the same corrected
        # input; this uses no wheel measurements.
        gpx_epoch_slope = 1.0
        gpx_epoch_intercept = embedded_epoch - float(frame_t[0])
        pipeline_sync_range = 0.0
    else:
        gpx_epoch_slope = slope
        gpx_epoch_intercept = intercept
        pipeline_sync_range = args.sync_range
    gpx_t = prepare_gpx(
        segment, gpx, gpx_epoch_slope, gpx_epoch_intercept,
        float(frame_t[0]), float(frame_t[-1]),
    )
    manifest = {
        "source_segment": str(segment),
        "video_frames": int(len(frame_t)),
        "video_fps": fps,
        "video_first_frame_boot_seconds": float(frame_t[0]),
        "video_last_frame_boot_seconds": float(frame_t[-1]),
        "true_video_start_epoch": true_epoch,
        "embedded_video_start_epoch": float(embedded_epoch),
        "known_metadata_rounding_error_seconds": true_epoch - embedded_epoch,
        "gnss_clock_fit_residual_std_seconds": clock_residual,
        "gpx_points": int(len(gpx_t)),
        "gpx_contains_reported_speed": False,
        "clock_mode": args.clock_mode,
        "pipeline_sync_range_seconds": pipeline_sync_range,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    script_dir = Path(__file__).resolve().parent
    run_pipeline(
        script_dir, video, gpx, dry_fit, True, args, pipeline_sync_range
    )
    run_pipeline(
        script_dir, video, gpx, full_fit, False, args, pipeline_sync_range
    )
    metrics = evaluate(segment, dry_fit, full_fit, manifest, output)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote validation artifacts to {output}")


if __name__ == "__main__":
    main()
