import unittest

import numpy as np

from speed_estimation import arithmetic_mean_scale, find_linear_offset, linear_correlation, split_fit_timestamp_shift, time_mean_scale


class MeanScalingTests(unittest.TestCase):
    def test_arithmetic_scaling_rejects_zero_motion_for_nonzero_target(self):
        with self.assertRaisesRegex(ValueError, "zero-motion"):
            arithmetic_mean_scale(np.zeros(3), 2.0)

    def test_time_scaling_rejects_zero_motion_for_nonzero_target(self):
        with self.assertRaisesRegex(ValueError, "zero-motion"):
            time_mean_scale(np.arange(3.0), np.zeros(3), 2.0)

    def test_zero_motion_is_valid_for_zero_target(self):
        result, factor = time_mean_scale(
            np.arange(3.0), np.zeros(3), 0.0
        )
        np.testing.assert_array_equal(result, np.zeros(3))
        self.assertEqual(factor, 1.0)


class LinearClockCorrelationTests(unittest.TestCase):
    def test_linear_correlation_retains_magnitude(self):
        self.assertGreater(
            linear_correlation([0, 0, 1, 1], [0, 0, 10, 10]),
            linear_correlation([0, 0, 1, 1], [10, 10, 0, 0]),
        )

    def test_linear_offset_recovers_static_shift(self):
        times = np.arange(50.0)
        motion = np.random.default_rng(7).normal(size=len(times))
        offset, score, _, _, at_limit = find_linear_offset(
            times, motion, times + 3, motion, 10,
        )
        self.assertAlmostEqual(offset, 3.0)
        self.assertGreater(score, 0.99)
        self.assertFalse(at_limit)


class TimestampShiftTests(unittest.TestCase):
    def test_fractional_shift_keeps_a_phase_for_garmin_speed_interpolation(self):
        encoded, phase = split_fit_timestamp_shift(8.55)
        self.assertEqual(encoded, -9)
        self.assertAlmostEqual(phase, -0.45)


if __name__ == "__main__":
    unittest.main()
