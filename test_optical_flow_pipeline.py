import unittest

import numpy as np

from optical_flow_pipeline import median_flow_magnitude, roi_bounds


class OpticalFlowPipelineTests(unittest.TestCase):
    def test_roi_dimensions(self):
        self.assertEqual(roi_bounds((360, 640)), (64, 72, 576, 306))

    def test_identical_frames_have_zero_median_motion(self):
        frame = np.arange(360 * 640, dtype=np.uint8).reshape(360, 640)
        self.assertAlmostEqual(median_flow_magnitude(frame, frame), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
