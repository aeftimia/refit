#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: insta360_video_speed_fit.sh [--dry-run] VIDEO.mp4 GARMIN.fit [OUTPUT.fit]

Options:
  --dry-run        Auto-sync and write original GPS-derived motion without
                   replacing it with optical-flow speeds. Every original FIT
                   message is retained.

Environment variables:
  VIDEO_TIMEZONE   Time zone used when MP4 metadata has no offset (default: UTC).
                   Examples: UTC, America/New_York, +02:00
  SAMPLE_FPS       Optical-flow samples per second (default: 4)
  FLOW_WORKERS     Parallel optical-flow workers (default: 16)
  SYNC_RANGE       Maximum automatic clock correction in seconds (default: 300;
                   set to 0 to disable automatic alignment)
EOF
}

dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=1
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *) break ;;
  esac
done

[[ $# -ge 2 && $# -le 3 ]] || { usage >&2; exit 2; }

video=$1
input_fit=$2
requested_output=${3:-"${video%.*}_speed.fit"}
case "$requested_output" in
  *.fit|*.FIT) output_fit=$requested_output ;;
  *) output_fit="${requested_output}.fit" ;;
esac
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_script="$script_dir/video_speed_fit.py"

for command in exiftool ffmpeg python3; do
  command -v "$command" >/dev/null || { echo "Error: $command is required." >&2; exit 127; }
done
[[ -f "$video" ]] || { echo "Error: video not found: $video" >&2; exit 2; }
[[ -f "$input_fit" ]] || { echo "Error: Garmin FIT not found: $input_fit" >&2; exit 2; }
[[ -f "$python_script" ]] || { echo "Error: processor not found: $python_script" >&2; exit 2; }
python3 -c 'import fit_tool' 2>/dev/null || {
  echo "Error: fit-tool is required. Install with: python3 -m pip install -r requirements.txt" >&2
  exit 127
}

# Ask exiftool for numeric duration and all likely QuickTime/Insta360 date tags.
metadata=$(exiftool -j -n -api QuickTimeUTC=1 \
  -Duration -CreateDate -MediaCreateDate -TrackCreateDate \
  -DateTimeOriginal -TimeZone -OffsetTimeOriginal "$video")

run_processor() {
  python3 "$python_script" \
    --video "$video" \
    --fit "$input_fit" \
    --output "$output_fit" \
    --metadata-json "$metadata" \
    --default-timezone "${VIDEO_TIMEZONE:-UTC}" \
    --flow-workers "${FLOW_WORKERS:-16}" \
    --sync-range "${SYNC_RANGE:-300}" \
    --sample-fps "${SAMPLE_FPS:-4}" \
    "$@"
}

if [[ "$dry_run" == 1 ]]; then
  run_processor --dry-run
else
  run_processor
fi

echo "Wrote $output_fit"
