i#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage:"
    echo "  $0 video.mp4 track.gpx [output.gpx]"
    exit 1
fi

VIDEO="$1"
GPX="$2"
OUTPUT="${3:-output.gpx}"

if [[ ! -f "$VIDEO" ]]; then
    echo "Video not found: $VIDEO"
    exit 1
fi

if [[ ! -f "$GPX" ]]; then
    echo "GPX not found: $GPX"
    exit 1
fi

exiftool -j -api QuickTimeUTC "$VIDEO" \
| python3 correct_speed.py \
        --video "$VIDEO" \
        --gpx "$GPX" \
        --output "$OUTPUT"
