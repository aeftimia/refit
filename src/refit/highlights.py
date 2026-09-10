"""Physically grounded highlight scoring from planar vehicle motion."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from .speed_estimation import EARTH_RADIUS_METRES

ScoreMode = Literal["lateral", "bivector", "total"]


@dataclass(frozen=True)
class GeometricMotion:
    """The scalar and bivector parts of velocity times acceleration."""

    times: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    scalar: np.ndarray
    bivector: np.ndarray
    lateral: np.ndarray
    total: np.ndarray

    def score(self, mode: ScoreMode) -> np.ndarray:
        if mode == "lateral":
            return self.lateral
        if mode == "bivector":
            return np.abs(self.bivector)
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

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_dict(self) -> dict:
        return {
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


def smooth_vectors(values, window_samples: int) -> np.ndarray:
    """Apply a centered moving average without shortening the series."""
    values = np.asarray(values, dtype=float)
    window_samples = int(window_samples)
    if window_samples <= 1:
        return values.copy()
    if window_samples % 2 == 0:
        window_samples += 1
    window_samples = min(window_samples, len(values) - (1 - len(values) % 2))
    if window_samples <= 1:
        return values.copy()
    radius = window_samples // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(window_samples, dtype=float) / window_samples
    return np.column_stack([
        np.convolve(padded[:, axis], kernel, mode="valid")
        for axis in range(values.shape[1])
    ])


def geometric_motion(
    times,
    velocity,
    *,
    smoothing_seconds: float = 1.0,
) -> GeometricMotion:
    """Compute the geometric product ``velocity * acceleration`` once.

    In the horizontal plane its scalar part is specific kinetic power and its
    signed bivector coefficient describes turning of the velocity vector.  The
    multivector norm is ``|velocity| * |acceleration|``.
    """
    times, velocity = _validate_motion(times, velocity)
    median_step = float(np.median(np.diff(times)))
    window_samples = max(1, round(float(smoothing_seconds) / median_step))
    velocity = smooth_vectors(velocity, window_samples)
    acceleration = np.gradient(velocity, times, axis=0)
    scalar = np.einsum("ij,ij->i", velocity, acceleration)
    bivector = velocity[:, 0] * acceleration[:, 1] - velocity[:, 1] * acceleration[:, 0]
    speed = np.linalg.norm(velocity, axis=1)
    lateral = np.divide(
        np.abs(bivector), speed,
        out=np.zeros_like(bivector), where=speed > 1e-9,
    )
    total = np.hypot(scalar, bivector)
    return GeometricMotion(
        times, velocity, acceleration, scalar, bivector, lateral, total,
    )


def geometric_motion_from_gps(
    times,
    latitude,
    longitude,
    *,
    speed=None,
    smoothing_seconds: float = 1.0,
) -> GeometricMotion:
    """Construct planar velocity vectors from a geographic track.

    GPS positions provide the travel direction.  When a recorded scalar speed
    is supplied, it provides the magnitude because it is normally less noisy
    than differentiating positions at approximately 1 Hz.
    """
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
    position_velocity = np.gradient(position, times, axis=0)

    if speed is None:
        velocity = position_velocity
    else:
        speed = np.asarray(speed, dtype=float)
        if speed.shape != times.shape:
            raise ValueError("speed must have the same shape as GPS times")
        valid = np.isfinite(speed)
        if valid.sum() < 2:
            raise ValueError("speed must contain at least two finite samples")
        speed = np.interp(times, times[valid], speed[valid])
        magnitude = np.linalg.norm(position_velocity, axis=1)
        direction = np.zeros_like(position_velocity)
        moving = magnitude > 1e-6
        direction[moving] = position_velocity[moving] / magnitude[moving, None]
        velocity = direction * np.maximum(speed, 0)[:, None]
    return geometric_motion(
        times, velocity, smoothing_seconds=smoothing_seconds,
    )


def _window_candidates(
    timeline: MotionTimeline,
    mode: ScoreMode,
    clip_duration: float,
) -> list[HighlightClip]:
    motion = timeline.motion
    scores = motion.score(mode)
    half = clip_duration / 2
    candidates = []
    for index, peak in enumerate(motion.times):
        start = max(float(motion.times[0]), float(peak - half))
        end = min(float(motion.times[-1]), start + clip_duration)
        start = max(float(motion.times[0]), end - clip_duration)
        inside = (motion.times >= start) & (motion.times <= end)
        if end <= start or not inside.any():
            continue
        # Mean intensity makes fixed-duration windows comparable while retaining
        # enough context around a sharp event to produce a watchable clip.
        window_score = float(np.mean(scores[inside]))
        peak_index = int(np.flatnonzero(inside)[np.argmax(scores[inside])])
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
        ))
    return candidates


def _overlaps(first: HighlightClip, second: HighlightClip) -> bool:
    return first.source == second.source and first.start < second.end and second.start < first.end


def select_highlights(
    timelines: Sequence[MotionTimeline],
    *,
    mode: ScoreMode,
    target_duration: float,
    clip_duration: float = 10.0,
    order: Literal["chronological", "interesting"] = "chronological",
) -> list[HighlightClip]:
    """Select the strongest non-overlapping windows up to a duration budget."""
    if target_duration <= 0 or clip_duration <= 0:
        raise ValueError("highlight and clip durations must be positive")
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
    order: Literal["chronological", "interesting"] = "chronological",
) -> dict[ScoreMode, list[HighlightClip]]:
    """Run both rankings through exactly the same selection machinery."""
    return {
        mode: select_highlights(
            timelines,
            mode=mode,
            target_duration=target_duration,
            clip_duration=clip_duration,
            order=order,
        )
        for mode in ("lateral", "bivector", "total")
    }


def comparison_manifests(
    timelines: Sequence[MotionTimeline],
    *,
    target_duration: float,
    clip_duration: float = 10.0,
    order: Literal["chronological", "interesting"] = "chronological",
) -> dict[ScoreMode, dict]:
    """Return comparable JSON-ready manifests from one geometric analysis."""
    selections = compare_highlight_modes(
        timelines,
        target_duration=target_duration,
        clip_duration=clip_duration,
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
    order: Literal["chronological", "interesting"] = "chronological",
) -> dict[ScoreMode, Path]:
    """Write sibling ``_bivector.json`` and ``_total.json`` manifests."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    manifests = comparison_manifests(
        timelines,
        target_duration=target_duration,
        clip_duration=clip_duration,
        order=order,
    )
    outputs = {}
    for mode, manifest in manifests.items():
        output = prefix.with_name(f"{prefix.name}_{mode}.json")
        output.write_text(json.dumps(manifest, indent=2) + "\n")
        outputs[mode] = output
    return outputs
