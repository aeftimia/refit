#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: insta360_video_speed_fit.sh VIDEO.mp4 [GARMIN.fit]

Downloads the matching Garmin activity when GARMIN.fit is omitted and writes
<video-stem>_speed.fit in the current directory.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi
[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }

video=$1
input_fit=${2:-}
video_name=${video##*/}
video_stem=${video_name%.*}
output_fit="$PWD/${video_stem}_speed.fit"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_script="$script_dir/video_speed_fit.py"
garmin_script="$script_dir/garmin_connect_fit.py"
python_bin="$script_dir/venv/bin/python"

for command in exiftool ffmpeg; do
  command -v "$command" >/dev/null || { echo "Error: $command is required." >&2; exit 127; }
done
[[ -x "$python_bin" ]] || {
  echo "Error: project virtual environment is missing: $python_bin" >&2
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
  [[ -f "$input_fit" ]] || { echo "Error: Garmin FIT not found: $input_fit" >&2; exit 2; }
else
  [[ -f "$garmin_script" ]] || { echo "Error: Garmin downloader not found: $garmin_script" >&2; exit 2; }
  input_fit=$("$python_bin" "$garmin_script" --metadata-json "$metadata")
fi

"$python_bin" "$python_script" \
  --video "$video" \
  --fit "$input_fit" \
  --output "$output_fit" \
  --metadata-json "$metadata"

echo "Wrote $output_fit"
