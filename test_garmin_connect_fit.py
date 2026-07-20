import io
import unittest
import zipfile
from datetime import datetime, timezone

from garmin_connect_fit import choose_activity, extract_fit


def fake_fit(size=20):
    payload = bytearray(max(size, 14))
    payload[0] = 14
    payload[8:12] = b".FIT"
    return bytes(payload)


class ActivitySelectionTests(unittest.TestCase):
    def test_prefers_activity_covering_most_of_video(self):
        start = datetime(2026, 7, 17, 14, 55, tzinfo=timezone.utc)
        end = datetime(2026, 7, 17, 15, 5, tzinfo=timezone.utc)
        activities = [
            {
                "activityId": 1,
                "activityName": "short overlap",
                "startTimeGMT": "2026-07-17 15:04:00",
                "elapsedDuration": 600,
            },
            {
                "activityId": 2,
                "activityName": "ride",
                "startTimeGMT": "2026-07-17 14:00:00",
                "elapsedDuration": 7200,
            },
        ]
        self.assertEqual(choose_activity(activities, start, end).activity_id, "2")

    def test_video_may_end_after_activity(self):
        start = datetime(2026, 7, 17, 14, 55, tzinfo=timezone.utc)
        end = datetime(2026, 7, 17, 15, 15, tzinfo=timezone.utc)
        activities = [
            {
                "activityId": 3,
                "startTimeGMT": "2026-07-17 14:30:00",
                "elapsedDuration": 2100,
            }
        ]
        selected = choose_activity(activities, start, end)
        self.assertEqual(selected.activity_id, "3")

    def test_rejects_unrelated_nearest_activity(self):
        start = datetime(2026, 7, 17, 14, 55, tzinfo=timezone.utc)
        end = datetime(2026, 7, 17, 15, 5, tzinfo=timezone.utc)
        activities = [
            {
                "activityId": 4,
                "startTimeGMT": "2026-07-17 18:00:00",
                "elapsedDuration": 3600,
            }
        ]
        with self.assertRaisesRegex(ValueError, "nearest activity"):
            choose_activity(activities, start, end)


class DownloadExtractionTests(unittest.TestCase):
    def test_accepts_direct_fit(self):
        payload = fake_fit()
        self.assertEqual(extract_fit(payload), payload)

    def test_extracts_largest_fit_from_original_archive(self):
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as archive:
            archive.writestr("settings.fit", fake_fit(20))
            archive.writestr("activity.fit", fake_fit(100))
        self.assertEqual(extract_fit(archive_bytes.getvalue()), fake_fit(100))

    def test_rejects_non_fit_download(self):
        with self.assertRaisesRegex(ValueError, "not a valid FIT"):
            extract_fit(b"not a fit")


if __name__ == "__main__":
    unittest.main()
