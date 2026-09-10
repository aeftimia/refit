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
python_bin="$script_dir/venv/bin/python"

for command in exiftool ffmpeg; do
  command -v "$command" >/dev/null || { echo "Error: $command is required." >&2; exit 127; }
done
[[ -x "$python_bin" ]] || {
  echo "Error: project virtual environment is missing: $python_bin" >&2
  exit 127
}
[[ -f "$video" ]] || { echo "Error: video not found: $video" >&2; exit 2; }
"$python_bin" -c 'import fit_tool' 2>/dev/null || {
  echo "Error: fit-tool is required. Install with: $python_bin -m pip install -r $script_dir/requirements.txt" >&2
  exit 127
}

args=("$video" --output "$output_fit")
[[ -z "$input_fit" ]] || args+=("$input_fit")
PYTHONPATH="$script_dir/src${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -m refit "${args[@]}"

echo "Wrote $output_fit"
