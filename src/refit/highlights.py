"""Physically grounded highlight scoring from planar vehicle motion."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, PchipInterpolator

from .speed_estimation import EARTH_RADIUS_METRES

ScoreMode = Literal["lateral", "bivector", "bivector_3d", "total"]
DEFAULT_SCORE_MODE: ScoreMode = "bivector"


@dataclass(frozen=True)
class GeometricMotion:
    """The scalar and bivector parts of velocity times acceleration."""

    times: np.ndarray
    distance: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    scalar: np.ndarray
    bivector: np.ndarray
    lateral: np.ndarray
    total: np.ndarray
    vertical_acceleration: np.ndarray | None = None

    def score(self, mode: ScoreMode = DEFAULT_SCORE_MODE) -> np.ndarray:
        if mode == "lateral":
            return self.lateral
        if mode == "bivector":
            return np.abs(self.bivector)
        if mode == "bivector_3d":
            if self.vertical_acceleration is None:
                raise ValueError("3D bivector scoring requires camera IMU data")
            speed = np.linalg.norm(self.velocity, axis=1)
            return np.hypot(self.bivector, speed * self.vertical_acceleration)
        if mode == "total":
            return self.total
        raise ValueError(f"unknown highlight score mode: {mode}")


@dataclass(frozen=True)
class MotionTimeline:
    """Motion samples expressed in seconds relative to one source video."""

    source: str
    motion: GeometricMotion


@dataclass(frozen=True)
class HighlightClip:
    source: str
    start: float
    end: float
    peak: float
    score: float
    scalar_at_peak: float
    bivector_at_peak: float
    lateral_at_peak: float
    total_at_peak: float
    vertical_acceleration_at_peak: float | None = None
    bivector_3d_at_peak: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_dict(self) -> dict:
        result = {
            "source": self.source,
            "start_seconds": self.start,
            "end_seconds": self.end,
            "duration_seconds": self.duration,
            "peak_seconds": self.peak,
            "score": self.score,
            "scalar_at_peak": self.scalar_at_peak,
            "bivector_at_peak": self.bivector_at_peak,
            "lateral_at_peak": self.lateral_at_peak,
            "total_at_peak": self.total_at_peak,
        }
        if self.vertical_acceleration_at_peak is not None:
            result["vertical_acceleration_at_peak"] = self.vertical_acceleration_at_peak
            result["bivector_3d_at_peak"] = self.bivector_3d_at_peak
        return result


def _validate_motion(times, velocity) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if times.ndim != 1 or velocity.shape != (len(times), 2):
        raise ValueError("velocity must have shape (len(times), 2)")
    if len(times) < 3:
        raise ValueError("at least three motion samples are required")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(velocity)):
        raise ValueError("motion samples must be finite")
    if np.any(np.diff(times) <= 0):
        raise ValueError("motion sample times must be strictly increasing")
    return times, velocity


def geometric_motion(times, velocity) -> GeometricMotion:
    """Compute the geometric product ``velocity * acceleration`` once.

    In the horizontal plane its scalar part is specific kinetic power and its
    signed bivector coefficient describes turning of the velocity vector.  The
    multivector norm is ``|velocity| * |acceleration|``.
    """
    times, velocity = _validate_motion(times, velocity)
    acceleration = np.gradient(velocity, times, axis=0)
    scalar = np.einsum("ij,ij->i", velocity, acceleration)
    bivector = velocity[:, 0] * acceleration[:, 1] - velocity[:, 1] * acceleration[:, 0]
    speed = np.linalg.norm(velocity, axis=1)
    lateral = np.divide(
        np.abs(bivector), speed,
        out=np.zeros_like(bivector), where=speed > 1e-9,
    )
    total = np.hypot(scalar, bivector)
    distance = cumulative_trapezoid(speed, times, initial=0)
    return GeometricMotion(
        times, distance, velocity, acceleration, scalar, bivector, lateral, total,
    )


def geometric_motion_from_gps(
    times,
    latitude,
    longitude,
    *,
    speed=None,
    speed_times=None,
    sample_times=None,
) -> GeometricMotion:
    """Derive motion analytically from a distance-parameterized GPS path."""
    times = np.asarray(times, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)
    if not (len(times) == len(latitude) == len(longitude)):
        raise ValueError("GPS arrays must have equal lengths")
    if len(times) < 3 or np.any(np.diff(times) <= 0):
        raise ValueError("GPS times must contain at least three increasing samples")
    latitude_radians = np.radians(latitude)
    longitude_radians = np.unwrap(np.radians(longitude))
    reference_latitude = float(np.mean(latitude_radians))
    east = EARTH_RADIUS_METRES * np.cos(reference_latitude) * (
        longitude_radians - longitude_radians[0]
    )
    north = EARTH_RADIUS_METRES * (latitude_radians - latitude_radians[0])
    position = np.column_stack((east, north))
    segment_distance = np.linalg.norm(np.diff(position, axis=0), axis=1)
    path_distance = np.r_[0.0, np.cumsum(segment_distance)]
    distinct = np.r_[True, np.diff(path_distance) > 1e-6]
    if distinct.sum() < 3:
        raise ValueError("GPS track must contain at least three distinct positions")
    path_distance = path_distance[distinct]
    position = position[distinct]
    position_times = times[distinct]

    east_spline = CubicSpline(path_distance, position[:, 0], bc_type="natural")
    north_spline = CubicSpline(path_distance, position[:, 1], bc_type="natural")
    distance_at_time = PchipInterpolator(position_times, path_distance)

    if speed is None:
        speed_times = position_times
        speed_values = np.maximum(distance_at_time.derivative()(speed_times), 0)
    else:
        speed_values = np.asarray(speed, dtype=float)
        speed_times = times if speed_times is None else np.asarray(speed_times, dtype=float)
        if speed_values.shape != speed_times.shape:
            raise ValueError("speed and speed_times must have the same shape")
        valid = np.isfinite(speed_values)
        if valid.sum() < 2:
            raise ValueError("speed must contain at least two finite samples")
        speed_times = speed_times[valid]
        speed_values = np.maximum(speed_values[valid], 0)
    speed_at_time = PchipInterpolator(speed_times, speed_values)

    if sample_times is None:
        lower = max(position_times[0], speed_times[0])
        upper = min(position_times[-1], speed_times[-1])
        sample_times = np.arange(lower, upper + 0.05, 0.1)
    else:
        sample_times = np.asarray(sample_times, dtype=float)
    if len(sample_times) < 3 or np.any(np.diff(sample_times) <= 0):
        raise ValueError("sample_times must contain at least three increasing values")

    distance = np.clip(
        distance_at_time(sample_times), path_distance[0], path_distance[-1]
    )
    first = np.column_stack((
        east_spline(distance, 1), north_spline(distance, 1),
    ))
    second = np.column_stack((
        east_spline(distance, 2), north_spline(distance, 2),
    ))
    first_norm = np.linalg.norm(first, axis=1)
    tangent = np.divide(
        first, first_norm[:, None],
        out=np.zeros_like(first), where=first_norm[:, None] > 1e-12,
    )
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    curvature = np.divide(
        first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0],
        first_norm ** 3,
        out=np.zeros_like(first_norm), where=first_norm > 1e-12,
    )
    sampled_speed = np.maximum(speed_at_time(sample_times), 0)
    longitudinal_acceleration = speed_at_time.derivative()(sample_times)
    lateral_acceleration = sampled_speed ** 2 * curvature
    velocity = sampled_speed[:, None] * tangent
    acceleration = (
        longitudinal_acceleration[:, None] * tangent
        + lateral_acceleration[:, None] * normal
    )
    scalar = sampled_speed * longitudinal_acceleration
    bivector = sampled_speed * lateral_acceleration
    lateral = np.abs(lateral_acceleration)
    total = np.hypot(scalar, bivector)
    return GeometricMotion(
        sample_times,
        distance - distance[0],
        velocity,
        acceleration,
        scalar,
        bivector,
        lateral,
        total,
    )


def _window_candidates(
    timeline: MotionTimeline,
    mode: ScoreMode,
    clip_duration: float,
) -> list[HighlightClip]:
    motion = timeline.motion
    scores = motion.score(mode)
    score_at_time = PchipInterpolator(motion.times, scores)
    time_integral = score_at_time.antiderivative()
    half = clip_duration / 2
    candidates = []
    for peak in motion.times:
        start = max(float(motion.times[0]), float(peak - half))
        end = min(float(motion.times[-1]), start + clip_duration)
        start = max(float(motion.times[0]), end - clip_duration)
        first_index = int(np.searchsorted(motion.times, start, side="left"))
        end_index = int(np.searchsorted(motion.times, end, side="right"))
        if end <= start or end_index <= first_index:
            continue
        window_score = float(
            (time_integral(end) - time_integral(start)) / (end - start)
        )
        peak_index = first_index + int(np.argmax(scores[first_index:end_index]))
        vertical_at_peak = (
            None if motion.vertical_acceleration is None
            else float(motion.vertical_acceleration[peak_index])
        )
        candidates.append(HighlightClip(
            timeline.source,
            start,
            end,
            float(motion.times[peak_index]),
            window_score,
            float(motion.scalar[peak_index]),
            float(motion.bivector[peak_index]),
            float(motion.lateral[peak_index]),
            float(motion.total[peak_index]),
            vertical_at_peak,
            None if vertical_at_peak is None else float(scores[peak_index]),
        ))
    return candidates


def _overlaps(first: HighlightClip, second: HighlightClip) -> bool:
    return first.source == second.source and first.start < second.end and second.start < first.end


def select_highlights(
    timelines: Sequence[MotionTimeline],
    *,
    target_duration: float,
    mode: ScoreMode = DEFAULT_SCORE_MODE,
    clip_duration: float = 10.0,
    max_clips_per_source: int | None = None,
    order: Literal["chronological", "interesting"] = "chronological",
) -> list[HighlightClip]:
    """Select the strongest non-overlapping windows up to a duration budget."""
    if target_duration <= 0 or clip_duration <= 0:
        raise ValueError("highlight and clip durations must be positive")
    if max_clips_per_source is not None and max_clips_per_source <= 0:
        raise ValueError("max clips per source must be positive")
    chosen: list[HighlightClip] = []

    def add_best_windows(duration: float, limit: int) -> None:
        candidates = [
            candidate
            for timeline in timelines
            for candidate in _window_candidates(timeline, mode, duration)
        ]
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        added = 0
        for candidate in candidates:
            if max_clips_per_source is not None and sum(
                existing.source == candidate.source for existing in chosen
            ) >= max_clips_per_source:
                continue
            if any(_overlaps(candidate, existing) for existing in chosen):
                continue
            chosen.append(candidate)
            added += 1
            if added == limit:
                return

    full_count = int(target_duration // clip_duration)
    if full_count:
        add_best_windows(clip_duration, full_count)
    remainder = target_duration - full_count * clip_duration
    if remainder > 1e-9:
        add_best_windows(remainder, 1)
    if order == "chronological":
        source_order = {
            timeline.source: index for index, timeline in enumerate(timelines)
        }
        chosen.sort(key=lambda candidate: (
            source_order[candidate.source], candidate.start,
        ))
    elif order != "interesting":
        raise ValueError(f"unknown highlight ordering: {order}")
    return chosen


def compare_highlight_modes(
    timelines: Sequence[MotionTimeline],
    *,
    target_duration: float,
    clip_duration: float = 10.0,
    max_clips_per_source: int | None = None,
    order: Literal["chronological", "interesting"] = "chronological",
) -> dict[ScoreMode, list[HighlightClip]]:
    """Run both rankings through exactly the same selection machinery."""
    return {
        mode: select_highlights(
            timelines,
            mode=mode,
            target_duration=target_duration,
            clip_duration=clip_duration,
            max_clips_per_source=max_clips_per_source,
            order=order,
        )
        for mode in ("lateral", "bivector", "total")
    }


def comparison_manifests(
    timelines: Sequence[MotionTimeline],
    *,
    target_duration: float,
    clip_duration: float = 10.0,
    max_clips_per_source: int | None = None,
    order: Literal["chronological", "interesting"] = "chronological",
) -> dict[ScoreMode, dict]:
    """Return comparable JSON-ready manifests from one geometric analysis."""
    selections = compare_highlight_modes(
        timelines,
        target_duration=target_duration,
        clip_duration=clip_duration,
        max_clips_per_source=max_clips_per_source,
        order=order,
    )
    descriptions = {
        "lateral": "unit-velocity bivector |(velocity / |velocity|) ∧ acceleration|",
        "bivector": "absolute bivector part |velocity ∧ acceleration|",
        "total": "geometric-product norm |velocity acceleration|",
    }
    return {
        mode: {
            "score_mode": mode,
            "score_description": descriptions[mode],
            "target_duration_seconds": float(target_duration),
            "selected_duration_seconds": sum(clip.duration for clip in clips),
            "clip_duration_seconds": float(clip_duration),
            "max_clips_per_source": max_clips_per_source,
            "order": order,
            "clips": [clip.as_dict() for clip in clips],
        }
        for mode, clips in selections.items()
    }


def write_comparison_manifests(
    output_prefix: str | Path,
    timelines: Sequence[MotionTimeline],
    *,
    target_duration: float,
    clip_duration: float = 10.0,
    max_clips_per_source: int | None = None,
    order: Literal["chronological", "interesting"] = "chronological",
) -> dict[ScoreMode, Path]:
    """Write sibling ``_bivector.json`` and ``_total.json`` manifests."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    manifests = comparison_manifests(
        timelines,
        target_duration=target_duration,
        clip_duration=clip_duration,
        max_clips_per_source=max_clips_per_source,
        order=order,
    )
    outputs = {}
    for mode, manifest in manifests.items():
        output = prefix.with_name(f"{prefix.name}_{mode}.json")
        output.write_text(json.dumps(manifest, indent=2) + "\n")
        outputs[mode] = output
    return outputs
