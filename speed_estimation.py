"""Shared scalar speed-series estimation and validation primitives."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_METRES = 6371008.8


def haversine_distance(first, second):
    """Great-circle distance in metres between two latitude/longitude pairs."""
    latitude = np.radians([first[0], second[0]])
    longitude = np.radians([first[1], second[1]])
    delta_latitude = latitude[1] - latitude[0]
    delta_longitude = longitude[1] - longitude[0]
    value = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(latitude[0]) * np.cos(latitude[1])
        * np.sin(delta_longitude / 2) ** 2
    )
    return float(EARTH_RADIUS_METRES * 2 * np.arcsin(np.sqrt(value)))


def haversine_distances(latitude, longitude):
    """Vectorized consecutive great-circle distances in metres."""
    latitude = np.radians(np.asarray(latitude, dtype=float))
    longitude = np.radians(np.asarray(longitude, dtype=float))
    delta_latitude = np.diff(latitude)
    delta_longitude = np.diff(longitude)
    value = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(latitude[:-1]) * np.cos(latitude[1:])
        * np.sin(delta_longitude / 2) ** 2
    )
    return EARTH_RADIUS_METRES * 2 * np.arcsin(np.sqrt(value))


def split_fit_timestamp_shift(clock_offset):
    """Split a clock correction into FIT seconds and a speed-sampling phase.

    Standard activity timestamps encode whole seconds. The returned phase is
    added to synthetic-signal sample times so their displayed video positions
    retain the original subsecond correction.
    """
    desired_shift = -float(clock_offset)
    encoded_seconds = round(desired_shift)
    sampling_phase = encoded_seconds - desired_shift
    return encoded_seconds, sampling_phase


def averaged_ranks(values):
    """Return zero-based ranks, assigning the average rank to tied values."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def rank_correlation(first, second):
    """Spearman correlation with averaged ties and a defined flat-series result."""
    first = averaged_ranks(first)
    second = averaged_ranks(second)
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return -1.0
    return float(np.corrcoef(first, second)[0, 1])


def find_rank_offset(
    motion_times,
    motion,
    reference_times,
    reference,
    search_range,
    *,
    coarse_step=0.5,
    refinement_step=0.05,
    minimum_samples=20,
    support_penalty=0.0,
):
    """Find the static time offset maximizing rank correlation."""
    motion_times = np.asarray(motion_times, dtype=float)
    motion = np.asarray(motion, dtype=float)
    reference_times = np.asarray(reference_times, dtype=float)
    reference = np.asarray(reference, dtype=float)

    def score(offset):
        query = motion_times + offset
        valid = (query >= reference_times[0]) & (query <= reference_times[-1])
        count = int(valid.sum())
        if count < minimum_samples:
            return -2.0, count
        value = rank_correlation(
            motion[valid], np.interp(query[valid], reference_times, reference)
        )
        value -= support_penalty * (1 - valid.mean())
        return value, count

    coarse = np.arange(-search_range, search_range + coarse_step / 2, coarse_step)
    coarse_scores = [score(float(offset))[0] for offset in coarse]
    best_coarse = float(coarse[int(np.argmax(coarse_scores))])
    refinement = np.arange(
        max(-search_range, best_coarse - coarse_step),
        min(search_range, best_coarse + coarse_step) + refinement_step / 2,
        refinement_step,
    )
    candidates = [(*score(float(offset)), float(offset)) for offset in refinement]
    best_score, count, best_offset = max(candidates)
    zero_score, _ = score(0.0)
    at_limit = abs(best_offset) >= search_range - refinement_step / 2
    return best_offset, best_score, zero_score, count, at_limit


def find_rank_affine(
    motion_times,
    motion,
    reference_times,
    reference,
    search_range,
    *,
    max_drift_ppm=5000.0,
    minimum_samples=20,
    support_penalty=0.0,
):
    """Fit offset plus linear clock-rate error by rank correlation.

    Returns the offset at the first motion sample and the fractional clock-rate
    error. A rate of 0.001 means the reference clock gains one second per 1,000
    seconds of video.
    """
    motion_times = np.asarray(motion_times, dtype=float)
    motion = np.asarray(motion, dtype=float)
    reference_times = np.asarray(reference_times, dtype=float)
    reference = np.asarray(reference, dtype=float)
    duration = float(motion_times[-1] - motion_times[0])
    if duration <= 0:
        raise ValueError("affine clock fitting requires increasing motion times")
    if max_drift_ppm < 0:
        raise ValueError("max_drift_ppm cannot be negative")

    static_offset, static_score, zero_score, _, static_at_limit = find_rank_offset(
        motion_times,
        motion,
        reference_times,
        reference,
        search_range,
        minimum_samples=minimum_samples,
        support_penalty=support_penalty,
    )
    if max_drift_ppm == 0:
        return (
            static_offset, 0.0, static_score, static_score, zero_score,
            int(len(motion_times)), static_at_limit, False,
        )

    midpoint = (motion_times[0] + motion_times[-1]) / 2
    position = (motion_times - midpoint) / duration
    max_total_drift = duration * max_drift_ppm / 1_000_000

    def score(center_offset, total_drift):
        if (
            abs(center_offset - total_drift / 2) > search_range
            or abs(center_offset + total_drift / 2) > search_range
        ):
            return -2.0, 0
        query = motion_times + center_offset + total_drift * position
        valid = (query >= reference_times[0]) & (query <= reference_times[-1])
        count = int(valid.sum())
        if count < minimum_samples:
            return -2.0, count
        value = rank_correlation(
            motion[valid], np.interp(query[valid], reference_times, reference)
        )
        value -= support_penalty * (1 - valid.mean())
        return value, count

    drift_step = min(0.25, max_total_drift) if max_total_drift else 0.25
    coarse_drifts = np.arange(
        -max_total_drift,
        max_total_drift + drift_step / 2,
        drift_step,
    )
    center_radius = max(1.0, max_total_drift / 2)
    coarse_centers = np.arange(
        max(-search_range, static_offset - center_radius),
        min(search_range, static_offset + center_radius) + 0.25,
        0.5,
    )
    coarse = [
        (*score(float(center), float(drift)), float(center), float(drift))
        for drift in coarse_drifts
        for center in coarse_centers
    ]
    _, _, best_center, best_drift = max(coarse)

    fine_centers = np.arange(
        max(-search_range, best_center - 0.5),
        min(search_range, best_center + 0.5) + 0.025,
        0.05,
    )
    fine_drifts = np.arange(
        max(-max_total_drift, best_drift - 0.25),
        min(max_total_drift, best_drift + 0.25) + 0.025,
        0.05,
    )
    candidates = [
        (*score(float(center), float(drift)), float(center), float(drift))
        for drift in fine_drifts
        for center in fine_centers
    ]
    best_score, count, best_center, best_drift = max(candidates)
    start_offset = best_center - best_drift / 2
    rate = best_drift / duration
    offset_at_limit = abs(best_center) >= search_range - 0.025
    drift_at_limit = (
        max_total_drift > 0
        and abs(best_drift) >= max_total_drift - 0.025
    )
    return (
        start_offset, rate, best_score, static_score, zero_score, count,
        offset_at_limit, drift_at_limit,
    )


def stationary_interval_baseline(
    times, motion, reference_speed, *, stationary_tolerance=0.0, min_duration=0.0,
):
    """Return the quietest median among observed stationary intervals.

    All three arrays share a time grid. No assumed stationary fraction is used;
    when no nonzero-duration interval satisfies the explicit reference-speed
    tolerance, the baseline is exactly zero.
    """
    times = np.asarray(times, dtype=float)
    motion = np.asarray(motion, dtype=float)
    reference_speed = np.asarray(reference_speed, dtype=float)
    if not (len(times) == len(motion) == len(reference_speed)):
        raise ValueError("baseline inputs must have equal lengths")
    stopped = np.abs(reference_speed) <= stationary_tolerance
    medians = []
    supporting_samples = 0
    start = None
    for index, is_stopped in enumerate(stopped):
        if is_stopped and start is None:
            start = index
        if start is not None and (not is_stopped or index == len(stopped) - 1):
            end = index if is_stopped else index - 1
            if end > start and times[end] - times[start] >= min_duration:
                medians.append(float(np.median(motion[start:end + 1])))
                supporting_samples += end - start + 1
            start = None
    if not medians:
        return 0.0, 0, 0
    return min(medians), len(medians), supporting_samples


def arithmetic_mean_scale(values, target_mean):
    """Scale a sampled series to an explicit arithmetic mean."""
    values = np.asarray(values, dtype=float)
    source_mean = float(np.mean(values))
    if source_mean <= 1e-12:
        if abs(target_mean) <= 1e-12:
            return values.copy(), 1.0
        raise ValueError("cannot scale a zero-motion series to a nonzero mean")
    factor = float(target_mean) / source_mean
    return values * factor, factor


def time_mean_scale(times, values, target_mean):
    """Scale an irregular series to an explicit trapezoidal time mean."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(times) != len(values) or len(times) < 2:
        raise ValueError("time-mean scaling requires matching nontrivial arrays")
    duration = float(times[-1] - times[0])
    if duration <= 0:
        raise ValueError("time-mean scaling requires increasing timestamps")
    source_mean = float(np.trapezoid(values, times) / duration)
    if source_mean <= 1e-12:
        if abs(target_mean) <= 1e-12:
            return values.copy(), 1.0
        raise ValueError("cannot scale a zero-motion series to a nonzero mean")
    factor = float(target_mean) / source_mean
    return values * factor, factor


def error_summary(estimate, truth, *, include_rank=False):
    """Return the shared pointwise error metrics used by validators."""
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    error = estimate - truth
    absolute = np.abs(error)
    result = {
        "mae_mps": float(np.mean(absolute)),
        "rmse_mps": float(np.sqrt(np.mean(error ** 2))),
        "bias_mps": float(np.mean(error)),
        "p95_absolute_error_mps": float(np.percentile(absolute, 95)),
        "max_absolute_error_mps": float(np.max(absolute)),
        "pearson_correlation": float(np.corrcoef(estimate, truth)[0, 1]),
    }
    if include_rank:
        result["spearman_rank_correlation"] = rank_correlation(estimate, truth)
    return result
