import unittest

import numpy as np

from speed_estimation import arithmetic_mean_scale, time_mean_scale


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


if __name__ == "__main__":
    unittest.main()
