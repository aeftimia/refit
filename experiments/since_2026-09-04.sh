#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

"$project_dir/venv/bin/refit-highlight-experiment" \
  "/Users/aeftimia/Movies/The Shed Labor Day Weekend" \
  --fit-dir /Users/aeftimia/Downloads \
  --since 2026-09-04 \
  --timezone America/New_York \
  --duration 10m \
  --clip-duration 20s \
  --order interesting \
  --encoder copy \
  --output-dir "$project_dir/highlight_exports"
