#!/usr/bin/env python3
"""Compatibility entry point; the pipeline now reads and writes FIT directly."""

from video_speed_fit import main


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
