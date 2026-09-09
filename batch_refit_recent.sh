#!/usr/bin/env bash
# Align every recent camera MP4 with its matching Garmin activity and export FITs.
set -euo pipefail

camera_dir=${1:-/Volumes/Untitled/DCIM/Camera01}
output_dir=${2:-"$HOME/Downloads"}
days=${REFIT_RECENT_DAYS:-5}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
refit="$script_dir/insta360_video_speed_fit.sh"

[[ -d "$camera_dir" ]] || { echo "Error: camera directory not found: $camera_dir" >&2; exit 2; }
[[ -d "$output_dir" ]] || { echo "Error: output directory not found: $output_dir" >&2; exit 2; }
[[ -x "$refit" ]] || { echo "Error: ReFit entrypoint not executable: $refit" >&2; exit 2; }

count=$(find "$camera_dir" -maxdepth 1 -type f -iname '*.mp4' -mtime "-$days" -print | wc -l | tr -d ' ')
(( count > 0 )) || { echo "No MP4s modified in the last $days days." >&2; exit 1; }

echo "Aligning $count video(s) modified in the last $days days."
i=0
while IFS= read -r video; do
  i=$((i + 1))
  printf '\n[%d/%d] %s\n' "$i" "$count" "${video##*/}"
  (
    cd "$output_dir"
    "$refit" "$video"
  )
done < <(find "$camera_dir" -maxdepth 1 -type f -iname '*.mp4' -mtime "-$days" -print | sort)

echo "\nDone. FIT files are in $output_dir"
