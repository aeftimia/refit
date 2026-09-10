import unittest

import numpy as np

from refit.optical_flow_pipeline import (
    flow_divergence, flow_measurements, median_flow_magnitude, roi_bounds,
    signed_radial_flow, spherical_divergence, spherical_forward_projection,
    spherical_geometry,
)


class OpticalFlowPipelineTests(unittest.TestCase):
    def test_roi_dimensions(self):
        self.assertEqual(roi_bounds((360, 640)), (64, 72, 576, 306))

    def test_identical_frames_have_zero_median_motion(self):
        frame = np.arange(360 * 640, dtype=np.uint8).reshape(360, 640)
        self.assertAlmostEqual(median_flow_magnitude(frame, frame), 0.0, places=6)

    def test_outward_flow_has_positive_radial_and_divergence(self):
        height, width = 100, 120
        y, x = np.mgrid[:height, :width]
        flow = np.empty((height, width, 2), dtype=float)
        flow[..., 0] = .1 * (x - (width - 1) / 2)
        flow[..., 1] = .1 * (y - (height - 1) / 2)
        measurements = flow_measurements(flow)
        self.assertGreater(measurements["radial"], 0)
        self.assertAlmostEqual(measurements["divergence"], .2, places=6)

    def test_tangential_flow_has_zero_radial_and_divergence(self):
        height, width = 100, 120
        y, x = np.mgrid[:height, :width]
        flow = np.empty((height, width, 2), dtype=float)
        flow[..., 0] = -(y - (height - 1) / 2)
        flow[..., 1] = x - (width - 1) / 2
        self.assertAlmostEqual(float(np.median(signed_radial_flow(flow))), 0.0, places=6)
        self.assertAlmostEqual(float(np.median(flow_divergence(flow))), 0.0, places=6)

    @staticmethod
    def _pixel_flow_from_tangent(vector, x_basis, y_basis):
        """Express a tangent sphere field in the image-coordinate basis."""
        g_xx = np.sum(x_basis * x_basis, axis=2)
        g_xy = np.sum(x_basis * y_basis, axis=2)
        g_yy = np.sum(y_basis * y_basis, axis=2)
        b_x = np.sum(vector * x_basis, axis=2)
        b_y = np.sum(vector * y_basis, axis=2)
        determinant = g_xx * g_yy - g_xy ** 2
        return np.dstack(((g_yy * b_x - g_xy * b_y) / determinant,
                          (g_xx * b_y - g_xy * b_x) / determinant))

    def test_spherical_divergence_rejects_rigid_rotation(self):
        rays, x_basis, y_basis, _ = spherical_geometry((100, 120))
        rotation = np.cross(np.array([.2, -.1, .3]), rays)
        flow = self._pixel_flow_from_tangent(rotation, x_basis, y_basis)
        # Boundary finite differences are less exact; the interior should be
        # numerically zero for every rigid rotation on the viewing sphere.
        divergence = spherical_divergence(flow)[2:-2, 2:-2]
        self.assertLess(float(np.max(np.abs(divergence))), 3e-4)

    def test_spherical_forward_projection_detects_forward_translation(self):
        rays, x_basis, y_basis, _ = spherical_geometry((100, 120))
        axis = np.zeros_like(rays)
        axis[..., 2] = 1
        translation = -(axis - rays * rays[..., 2:3])
        flow = self._pixel_flow_from_tangent(translation, x_basis, y_basis)
        self.assertGreater(float(np.median(spherical_forward_projection(flow))), 0)


if __name__ == "__main__":
    unittest.main()
