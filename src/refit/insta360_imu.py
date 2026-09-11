"""Read and reduce high-rate Insta360 accelerometer trailer data."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfiltfilt


INSTA360_MAGIC = b"8db42d694ccc418790edff439fe026bf"
STANDARD_GRAVITY = 9.80665


def accelerometer_samples(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return video-relative seconds and camera-frame acceleration in g."""
    path = Path(path)
    with path.open("rb") as stream:
        stream.seek(0, 2)
        end = stream.tell()
        if end < 78:
            raise ValueError(f"file has no Insta360 trailer: {path}")
        stream.seek(end - 78)
        footer = stream.read(78)
        if footer[-32:] != INSTA360_MAGIC:
            raise ValueError(f"file has no Insta360 trailer: {path}")
        trailer_size = struct.unpack_from("<I", footer, 38)[0]
        directory_id, directory_size = struct.unpack_from("<HI", footer, 0)
        if directory_id != 0 or directory_size % 10:
            raise ValueError(f"invalid Insta360 trailer directory: {path}")
        stream.seek(end - 78 - directory_size)
        directory = stream.read(directory_size)
        entries = {
            record_id: (size, offset)
            for record_id, size, offset in (
                struct.unpack_from("<HII", directory, index)
                for index in range(0, directory_size, 10)
            )
            if record_id
        }
        if 3 not in entries:
            raise ValueError(f"Insta360 trailer has no accelerometer data: {path}")
        trailer_start = end - trailer_size
        size, offset = entries[3]
        stream.seek(trailer_start + offset)
        payload = stream.read(size)
        if size % 20:
            raise ValueError(f"unsupported Insta360 accelerometer layout: {path}")
        if 4 in entries:
            _, exposure_offset = entries[4]
            stream.seek(trailer_start + exposure_offset)
            video_clock_origin = struct.unpack("<Q", stream.read(8))[0]
        else:
            video_clock_origin = struct.unpack_from("<Q", payload, 0)[0]

    dtype = np.dtype([("clock", "<u8"), ("values", "<u2", (6,))])
    records = np.frombuffer(payload, dtype=dtype)
    times = (records["clock"].astype(float) - video_clock_origin) / 1_000_000
    acceleration = (records["values"][:, :3].astype(float) - 0x8000) / 1000
    valid = (
        np.isfinite(times)
        & np.all(np.isfinite(acceleration), axis=1)
        & np.all(np.abs(acceleration) < 20, axis=1)
    )
    return times[valid], acceleration[valid]


def vertical_acceleration_envelope(
    times: np.ndarray,
    acceleration_g: np.ndarray,
    sample_times: np.ndarray,
    *,
    gravity_cutoff_hz: float = 1.0,
    envelope_seconds: float = 0.2,
) -> np.ndarray:
    """Estimate orientation-independent vertical impact acceleration in m/s².

    Samples are reduced to 100 Hz, a slow gravity vector is estimated in the
    camera frame, and the residual parallel to gravity is converted to a short
    RMS envelope. This captures freefall/landing and suspension impacts without
    requiring a fixed camera orientation.
    """
    bins = np.floor(np.asarray(times) * 100).astype(np.int64)
    unique, first, counts = np.unique(bins, return_index=True, return_counts=True)
    reduced = np.vstack([
        np.median(acceleration_g[index:index + count], axis=0)
        for index, count in zip(first, counts)
    ])
    reduced_times = (unique + 0.5) / 100
    if len(reduced_times) < 20:
        raise ValueError("accelerometer recording is too short")
    gravity = sosfiltfilt(
        butter(2, gravity_cutoff_hz, btype="low", fs=100, output="sos"),
        reduced,
        axis=0,
    )
    gravity_norm = np.linalg.norm(gravity, axis=1)
    gravity_direction = np.divide(
        gravity,
        gravity_norm[:, None],
        out=np.zeros_like(gravity),
        where=gravity_norm[:, None] > 1e-9,
    )
    vertical = STANDARD_GRAVITY * np.sum(
        (reduced - gravity) * gravity_direction, axis=1
    )
    width = max(1, round(envelope_seconds * 100))
    squared = np.pad(vertical * vertical, (width // 2, width - 1 - width // 2), mode="edge")
    envelope = np.sqrt(np.convolve(squared, np.ones(width) / width, mode="valid"))
    return np.maximum(PchipInterpolator(reduced_times, envelope)(sample_times), 0)

