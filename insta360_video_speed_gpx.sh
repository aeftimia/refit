#!/usr/bin/env bash
set -euo pipefail

echo "Warning: insta360_video_speed_gpx.sh was renamed to insta360_video_speed_fit.sh" >&2
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/insta360_video_speed_fit.sh" "$@"
