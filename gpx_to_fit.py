#!/usr/bin/env python3
"""Convert a processed GPX into a minimal FIT activity for Insta360 Studio."""

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

from fit_tool.fit_file_builder import FitFileBuilder
from fit_tool.profile.messages.event_message import EventMessage
from fit_tool.profile.messages.file_id_message import FileIdMessage
from fit_tool.profile.messages.record_message import RecordMessage
from fit_tool.profile.profile_type import Event, EventType, FileType, Manufacturer


def local(tag):
    return tag.rsplit("}", 1)[-1]


def value(point, name):
    node = next((x for x in point.iter() if local(x.tag).lower() == name), None)
    return node.text if node is not None else None


def timestamp_ms(text):
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return round(dt.astimezone(timezone.utc).timestamp() * 1000)


def haversine(a, b):
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371008.8 * 2 * math.asin(math.sqrt(h))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_gpx")
    parser.add_argument("output_fit")
    args = parser.parse_args()

    root = ET.parse(args.input_gpx).getroot()
    points = [x for x in root.iter() if local(x.tag) == "trkpt"]
    if len(points) < 2:
        raise SystemExit("GPX must contain at least two trackpoints")

    coordinates = [(float(x.get("lat")), float(x.get("lon"))) for x in points]
    timestamps = [timestamp_ms(value(x, "time")) for x in points]
    interval_speeds = []
    for a, b, ta, tb in zip(coordinates, coordinates[1:], timestamps, timestamps[1:]):
        seconds = (tb - ta) / 1000
        interval_speeds.append(haversine(a, b) / seconds if seconds > 0 else 0.0)
    derived_speeds = [interval_speeds[0]]
    derived_speeds.extend((a + b) / 2 for a, b in zip(interval_speeds, interval_speeds[1:]))
    derived_speeds.append(interval_speeds[-1])

    builder = FitFileBuilder(auto_define=True)
    start = timestamp_ms(value(points[0], "time"))
    end = timestamp_ms(value(points[-1], "time"))

    file_id = FileIdMessage()
    file_id.type = FileType.ACTIVITY
    file_id.manufacturer = Manufacturer.GARMIN.value
    file_id.product = 0
    file_id.serial_number = 0
    file_id.time_created = start
    builder.add(file_id)

    event = EventMessage()
    event.event = Event.TIMER
    event.event_type = EventType.START
    event.timestamp = start
    builder.add(event)

    distance = 0.0
    previous = None
    for point, coordinate, derived_speed in zip(points, coordinates, derived_speeds):
        if previous is not None:
            distance += haversine(previous, coordinate)
        record = RecordMessage()
        record.timestamp = timestamp_ms(value(point, "time"))
        record.position_lat = coordinate[0]
        record.position_long = coordinate[1]
        record.distance = distance
        elevation = value(point, "ele")
        heart_rate = value(point, "hr")
        speed = value(point, "speed")
        if elevation is not None:
            record.enhanced_altitude = float(elevation)
        if heart_rate is not None:
            record.heart_rate = round(float(heart_rate))
        record.enhanced_speed = float(speed) if speed is not None else derived_speed
        builder.add(record)
        previous = coordinate

    event = EventMessage()
    event.event = Event.TIMER
    event.event_type = EventType.STOP
    event.timestamp = end
    builder.add(event)

    Path(args.output_fit).parent.mkdir(parents=True, exist_ok=True)
    builder.build().to_file(args.output_fit)
    print(f"Wrote {args.output_fit}: {len(points)} records, {distance:.1f} m")


if __name__ == "__main__":
    main()
