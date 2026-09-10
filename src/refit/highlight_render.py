"""Render an ordered highlight manifest as a review video."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def render_manifest(
    manifest_path: Path,
    output: Path,
    *,
    height: int | None = None,
    encoder: str = "libx264",
) -> None:
    manifest = json.loads(manifest_path.read_text())
    clips = manifest.get("clips", [])
    if not clips:
        raise ValueError("manifest contains no clips")
    if encoder == "copy" and height is not None:
        raise ValueError("stream-copy rendering cannot resize video")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="refit-highlights-") as directory:
        temporary = Path(directory)
        segments = []
        for index, clip in enumerate(clips, 1):
            source = Path(clip["source"])
            if not source.is_file():
                raise FileNotFoundError(f"highlight source not found: {source}")
            segment = temporary / f"segment-{index:04d}.mp4"
            command = [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{float(clip['start_seconds']):.6f}",
                "-i", str(source),
                "-t", f"{float(clip['duration_seconds']):.6f}",
                "-map", "0:v:0", "-map", "0:a:0?",
            ]
            if encoder == "copy":
                command.extend(["-c", "copy"])
            else:
                if height is not None:
                    command.extend(["-vf", f"scale=-2:{height}:flags=lanczos"])
                command.extend(["-c:v", encoder])
                if encoder == "h264_videotoolbox":
                    command.extend(["-b:v", "40M"])
                elif encoder == "hevc_videotoolbox":
                    command.extend(["-b:v", "40M", "-tag:v", "hvc1"])
                else:
                    command.extend(["-preset", "veryfast", "-crf", "20"])
                command.extend([
                    "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
                ])
            command.extend(["-movflags", "+faststart", str(segment)])
            print(f"Rendering segment {index}/{len(clips)}: {source.name}", flush=True)
            subprocess.run(command, check=True)
            segments.append(segment)

        concat_file = temporary / "segments.txt"
        concat_file.write_text("".join(
            f"file '{str(segment).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n"
            for segment in segments
        ))
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", "-movflags", "+faststart", str(output),
        ], check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="refit-render-highlights")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--height", type=int,
        help="optional output height; by default preserve the source resolution",
    )
    parser.add_argument(
        "--encoder", default="libx264",
        help="video encoder, or 'copy' to preserve original encoded streams",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        render_manifest(
            args.manifest.expanduser().resolve(),
            args.output.expanduser().resolve(),
            height=args.height,
            encoder=args.encoder,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Wrote {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
