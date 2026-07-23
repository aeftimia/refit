#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: insta360_video_speed_fit.sh [OPTIONS] VIDEO.mp4 [GARMIN.fit] [OUTPUT.fit]

Options:
  --dry-run        Explicitly select the default: auto-sync while preserving
                   original Garmin speeds and every FIT message.
  --full           Replace speeds with a fresh optical-flow estimate after
                   auto-syncing. This is slower and must be requested explicitly.
  --activity-id ID Download a specific Garmin Connect activity instead of
                   selecting the activity that best overlaps the video.
  --token-store DIR
                   Garmin authorization cache (default: ~/.garminconnect).
  -o, --output FIT Output path. Useful when GARMIN.fit is downloaded automatically.

Environment variables:
  REFIT_PYTHON     Python interpreter to use. By default, the repository's
                   venv/bin/python is preferred when available.
  VIDEO_TIMEZONE   Time zone used when MP4 metadata has no offset (default: UTC).
                   Examples: UTC, America/New_York, +02:00
  SAMPLE_FPS       Full-run optical-flow samples per second (default: 4).
                   Dry alignment uses up to 1,000 uniform adjacent-frame pairs.
  FLOW_WORKERS     Parallel optical-flow workers (default: 16)
  SYNC_RANGE       Maximum automatic clock correction in seconds (default: 300;
                   set to 0 to disable automatic alignment)
EOF
}

full_run=0
activity_id=
token_store=${GARMIN_TOKEN_STORE:-~/.garminconnect}
requested_output=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      full_run=0
      shift
      ;;
    --full)
      full_run=1
      shift
      ;;
    --activity-id)
      [[ $# -ge 2 ]] || { echo "Error: --activity-id requires a value" >&2; exit 2; }
      activity_id=$2
      shift 2
      ;;
    --token-store)
      [[ $# -ge 2 ]] || { echo "Error: --token-store requires a directory" >&2; exit 2; }
      token_store=$2
      shift 2
      ;;
    -o|--output)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a path" >&2; exit 2; }
      requested_output=$2
      shift 2
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

[[ $# -ge 1 && $# -le 3 ]] || { usage >&2; exit 2; }

video=$1
input_fit=${2:-}
if [[ $# -eq 3 ]]; then
  [[ -z "$requested_output" ]] || {
    echo "Error: output supplied both positionally and with --output" >&2
    exit 2
  }
  requested_output=$3
fi
video_name=${video##*/}
video_stem=${video_name%.*}
requested_output=${requested_output:-"$PWD/${video_stem}_speed.fit"}
case "$requested_output" in
  *.fit|*.FIT) output_fit=$requested_output ;;
  *) output_fit="${requested_output}.fit" ;;
esac
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_script="$script_dir/video_speed_fit.py"
garmin_script="$script_dir/garmin_connect_fit.py"
if [[ -n "${REFIT_PYTHON:-}" ]]; then
  python_bin=$REFIT_PYTHON
elif [[ -x "$script_dir/venv/bin/python" ]]; then
  python_bin="$script_dir/venv/bin/python"
else
  python_bin=python3
fi

for command in exiftool ffmpeg; do
  command -v "$command" >/dev/null || { echo "Error: $command is required." >&2; exit 127; }
done
command -v "$python_bin" >/dev/null || {
  echo "Error: Python interpreter not found: $python_bin" >&2
  exit 127
}
[[ -f "$video" ]] || { echo "Error: video not found: $video" >&2; exit 2; }
[[ -f "$python_script" ]] || { echo "Error: processor not found: $python_script" >&2; exit 2; }
"$python_bin" -c 'import fit_tool' 2>/dev/null || {
  echo "Error: fit-tool is required. Install with: $python_bin -m pip install -r $script_dir/requirements.txt" >&2
  exit 127
}

# Ask exiftool for numeric duration and all likely QuickTime/Insta360 date tags.
metadata=$(exiftool -j -n -api QuickTimeUTC=1 \
  -Duration -CreateDate -MediaCreateDate -TrackCreateDate \
  -DateTimeOriginal -TimeZone -OffsetTimeOriginal "$video")

if [[ -n "$input_fit" ]]; then
  [[ -z "$activity_id" ]] || {
    echo "Error: --activity-id cannot be combined with a local Garmin FIT" >&2
    exit 2
  }
  [[ -f "$input_fit" ]] || { echo "Error: Garmin FIT not found: $input_fit" >&2; exit 2; }
else
  [[ -f "$garmin_script" ]] || { echo "Error: Garmin downloader not found: $garmin_script" >&2; exit 2; }
  garmin_args=(
    --metadata-json "$metadata"
    --default-timezone "${VIDEO_TIMEZONE:-UTC}"
    --token-store "$token_store"
    --max-gap "${SYNC_RANGE:-300}"
  )
  if [[ -n "$activity_id" ]]; then
    garmin_args+=(--activity-id "$activity_id")
  fi
  input_fit=$("$python_bin" "$garmin_script" "${garmin_args[@]}")
fi

run_processor() {
  "$python_bin" "$python_script" \
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

if [[ "$full_run" == 1 ]]; then
  run_processor --full
else
  run_processor --dry-run
fi

echo "Wrote $output_fit"
