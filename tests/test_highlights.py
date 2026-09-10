import unittest
import tempfile
from pathlib import Path

import numpy as np

from refit.speed_estimation import EARTH_RADIUS_METRES
from refit.highlights import (
    MotionTimeline,
    compare_highlight_modes,
    geometric_motion,
    geometric_motion_from_gps,
    select_highlights,
    write_comparison_manifests,
)
from refit.highlight_cli import aligned_fit_path, discover_videos


class GeometricMotionTests(unittest.TestCase):
    def test_straight_acceleration_is_scalar_only(self):
        times = np.arange(5.0)
        velocity = np.column_stack((times, np.zeros_like(times)))
        motion = geometric_motion(times, velocity)
        np.testing.assert_allclose(motion.scalar, times)
        np.testing.assert_allclose(motion.bivector, 0)
        np.testing.assert_allclose(motion.total, times)

    def test_constant_speed_turn_is_bivector_only(self):
        times = np.linspace(0, 1, 101)
        velocity = np.column_stack((np.cos(times), np.sin(times)))
        motion = geometric_motion(times, velocity)
        np.testing.assert_allclose(motion.scalar[1:-1], 0, atol=5e-5)
        np.testing.assert_allclose(motion.bivector[1:-1], 1, atol=5e-5)
        np.testing.assert_allclose(motion.lateral[1:-1], 1, atol=5e-5)
        np.testing.assert_allclose(motion.total[1:-1], 1, atol=5e-5)

    def test_total_norm_contains_scalar_and_bivector(self):
        times = np.linspace(0, 1, 101)
        speed = np.exp(times)
        velocity = speed[:, None] * np.column_stack((np.cos(times), np.sin(times)))
        motion = geometric_motion(times, velocity)
        np.testing.assert_allclose(
            motion.total[1:-1],
            np.hypot(motion.scalar[1:-1], motion.bivector[1:-1]),
        )
        np.testing.assert_array_equal(motion.score(), motion.total)

    def test_gps_track_uses_recorded_speed_magnitude(self):
        times = np.arange(5.0)
        latitude = np.zeros(5)
        longitude = np.linspace(0, 0.001, 5)
        speed = np.full(5, 7.0)
        motion = geometric_motion_from_gps(
            times, latitude, longitude, speed=speed,
        )
        np.testing.assert_allclose(np.linalg.norm(motion.velocity, axis=1), 7.0)

    def test_distance_spline_recovers_constant_radius_turn(self):
        radius = 20.0
        speed = 5.0
        angle = np.linspace(0, np.pi / 2, 17)
        east = radius * np.sin(angle)
        north = radius * (1 - np.cos(angle))
        latitude = np.degrees(north / EARTH_RADIUS_METRES)
        longitude = np.degrees(east / EARTH_RADIUS_METRES)
        times = radius * angle / speed
        sample_times = np.linspace(times[1], times[-2], 100)
        motion = geometric_motion_from_gps(
            times,
            latitude,
            longitude,
            speed=np.full_like(times, speed),
            sample_times=sample_times,
        )
        np.testing.assert_allclose(
            np.median(motion.lateral), speed ** 2 / radius, rtol=0.08,
        )


class HighlightComparisonTests(unittest.TestCase):
    def test_modes_share_selection_but_can_choose_different_windows(self):
        times = np.arange(30.0)
        velocity = np.zeros((30, 2))
        velocity[:10, 0] = np.arange(10.0)  # straight acceleration
        angle = np.linspace(0, np.pi, 20)
        velocity[10:, 0] = 6 * np.cos(angle)
        velocity[10:, 1] = 6 * np.sin(angle)
        timeline = MotionTimeline(
            "ride.mp4",
            geometric_motion(times, velocity),
        )
        result = compare_highlight_modes(
            [timeline], target_duration=5, clip_duration=5, order="interesting"
        )
        self.assertEqual(len(result["bivector"]), 1)
        self.assertEqual(len(result["lateral"]), 1)
        self.assertEqual(len(result["total"]), 1)
        self.assertNotEqual(
            result["bivector"][0].peak,
            result["total"][0].peak,
        )

    def test_writes_three_named_manifests_from_one_timeline(self):
        times = np.arange(20.0)
        velocity = np.column_stack((times, np.sin(times)))
        timeline = MotionTimeline(
            "ride.mp4", geometric_motion(times, velocity)
        )
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_comparison_manifests(
                Path(directory) / "highlights",
                [timeline],
                target_duration=10,
                clip_duration=5,
            )
            self.assertEqual(outputs["bivector"].name, "highlights_bivector.json")
            self.assertEqual(outputs["lateral"].name, "highlights_lateral.json")
            self.assertEqual(outputs["total"].name, "highlights_total.json")
            self.assertTrue(outputs["bivector"].is_file())
            self.assertTrue(outputs["lateral"].is_file())
            self.assertTrue(outputs["total"].is_file())

    def test_selection_uses_remainder_to_reach_requested_duration(self):
        times = np.arange(30.0)
        velocity = np.column_stack((times, np.sin(times)))
        result = compare_highlight_modes(
            [MotionTimeline(
                "ride.mp4",
                geometric_motion(times, velocity),
            )],
            target_duration=12,
            clip_duration=5,
        )
        for clips in result.values():
            self.assertAlmostEqual(sum(clip.duration for clip in clips), 12)

    def test_selection_never_overlaps_clips_from_the_same_video(self):
        times = np.arange(100.0)
        velocity = np.column_stack((10 + np.sin(times), np.cos(times / 3)))
        result = compare_highlight_modes(
            [MotionTimeline(
                "ride.mp4",
                geometric_motion(times, velocity),
            )],
            target_duration=50,
            clip_duration=10,
        )
        for clips in result.values():
            chronological = sorted(clips, key=lambda clip: clip.start)
            self.assertTrue(all(
                first.end <= second.start
                for first, second in zip(chronological, chronological[1:])
            ))

    def test_video_discovery_ignores_macos_resource_forks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ride.mp4").touch()
            (root / "._ride.mp4").touch()
            self.assertEqual(
                discover_videos([root], None), [(root / "ride.mp4").resolve()]
            )

    def test_selection_caps_clips_per_source(self):
        times = np.arange(101.0)
        velocity = np.column_stack((times, np.zeros_like(times)))
        timelines = [
            MotionTimeline(name, geometric_motion(times, velocity))
            for name in ("first.mp4", "second.mp4")
        ]
        clips = select_highlights(
            timelines, target_duration=40, clip_duration=10,
            max_clips_per_source=2, order="interesting",
        )
        self.assertEqual(len(clips), 4)
        for source in ("first.mp4", "second.mp4"):
            self.assertEqual(sum(clip.source == source for clip in clips), 2)

    def test_exported_video_matches_sidecar_by_recording_time(self):
        metadata = {
            "CreateDate": "2026:09:06 09:11:13-04:00",
            "Duration": 30,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "VID_20260906_091113_031_speed.fit"
            expected.touch()
            self.assertEqual(
                aligned_fit_path(root / "boundary line 2.mp4", root, metadata),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
