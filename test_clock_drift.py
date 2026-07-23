import unittest

import numpy as np

from speed_estimation import find_rank_affine


class AffineClockTests(unittest.TestCase):
    def test_recovers_offset_and_linear_rate(self):
        relative = np.arange(0.0, 601.0)
        video_epoch = 1_700_000_000.0
        motion_times = video_epoch + relative
        motion = (
            np.sin(relative / 17)
            + .4 * np.sin(relative / 5.3)
            + .002 * relative
        )
        expected_offset = 12.4
        expected_rate = 1.0 / 600.0
        reference_times = video_epoch + np.arange(-30.0, 660.0)
        video_relative = (
            reference_times - video_epoch - expected_offset
        ) / (1 + expected_rate)
        reference = (
            np.sin(video_relative / 17)
            + .4 * np.sin(video_relative / 5.3)
            + .002 * video_relative
        )

        offset, rate, score, _, _, _, _, drift_limit = find_rank_affine(
            motion_times,
            motion,
            reference_times,
            reference,
            30,
            max_drift_ppm=3000,
            minimum_samples=100,
        )
        self.assertAlmostEqual(offset, expected_offset, delta=.15)
        self.assertAlmostEqual(rate, expected_rate, delta=2e-4)
        self.assertGreater(score, .99)
        self.assertFalse(drift_limit)


if __name__ == "__main__":
    unittest.main()
