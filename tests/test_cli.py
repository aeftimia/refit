import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from refit.cli import run


class CliTests(unittest.TestCase):
    def test_run_uses_default_output_and_calls_library_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "ride.mp4"
            fit = root / "activity.fit"
            video.touch()
            fit.touch()
            args = argparse.Namespace(
                video=video,
                fit=fit,
                output=None,
                activity_id=None,
                token_store=Path("~/.garminconnect"),
                cache_dir=Path("~/.cache/refit/garmin"),
                max_gap=45.0,
                clock_offset=None,
            )
            metadata = {"Duration": 10, "CreateDate": "2026-01-01T00:00:00+00:00"}
            with (
                patch("refit.cli.read_video_metadata", return_value=metadata),
                patch("refit.cli.align_video") as align,
                patch("refit.cli.Path.cwd", return_value=root),
            ):
                output = run(args)

            self.assertEqual(output, root / "ride_speed.fit")
            align.assert_called_once_with(
                str(video.resolve()), str(fit.resolve()), str(output), metadata,
                clock_offset=None,
            )


if __name__ == "__main__":
    unittest.main()
