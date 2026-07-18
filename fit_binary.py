"""Lossless fixed-width editing for FIT files with newer unknown messages."""

from dataclasses import dataclass
from pathlib import Path
import struct

from fit_tool.utils.crc import crc16


FIT_EPOCH_UNIX = 631065600


@dataclass
class FieldLocation:
    field_id: int
    size: int
    base_type: int
    offset: int


@dataclass
class DataLocation:
    global_id: int
    fields: dict[int, FieldLocation]


@dataclass
class TrackPoint:
    location: DataLocation
    timestamp: int
    position_lat: float
    position_long: float
    enhanced_speed: float | None


class FitBinary:
    """Patch selected scalar fields without decoding or rebuilding messages."""

    def __init__(self, path):
        self.source = Path(path)
        self.data = bytearray(self.source.read_bytes())
        self.header_size = self.data[0]
        self.records_size = struct.unpack_from("<I", self.data, 4)[0]
        self.records_end = self.header_size + self.records_size
        self.messages = self._scan()

    def _scan(self):
        definitions = {}
        messages = []
        cursor = self.header_size
        while cursor < self.records_end:
            header = self.data[cursor]
            cursor += 1
            if header & 0x80:
                raise ValueError("Compressed timestamp FIT records are not supported")
            local_id = header & 0x0F
            if header & 0x40:
                has_developer_fields = bool(header & 0x20)
                architecture = self.data[cursor + 1]
                endian = "little" if architecture == 0 else "big"
                global_id = int.from_bytes(self.data[cursor + 2:cursor + 4], endian)
                field_count = self.data[cursor + 4]
                cursor += 5
                fields = []
                for _ in range(field_count):
                    field_id, size, base_type = self.data[cursor:cursor + 3]
                    fields.append((field_id, size, base_type))
                    cursor += 3
                developer_fields = []
                if has_developer_fields:
                    developer_count = self.data[cursor]
                    cursor += 1
                    for _ in range(developer_count):
                        field_id, size, developer_index = self.data[cursor:cursor + 3]
                        developer_fields.append((field_id, size, developer_index))
                        cursor += 3
                definitions[local_id] = (global_id, fields, developer_fields)
                continue

            global_id, definition, developer_definition = definitions[local_id]
            field_locations = {}
            for field_id, size, base_type in definition:
                field_locations[field_id] = FieldLocation(field_id, size, base_type, cursor)
                cursor += size
            cursor += sum(size for _, size, _ in developer_definition)
            messages.append(DataLocation(global_id, field_locations))
        if cursor != self.records_end:
            raise ValueError("FIT record scan did not end at the declared data boundary")
        return messages

    def scalar(self, field, signed=False):
        return int.from_bytes(
            self.data[field.offset:field.offset + field.size], "little", signed=signed
        )

    def set_scalar(self, field, value, signed=False):
        self.data[field.offset:field.offset + field.size] = int(value).to_bytes(
            field.size, "little", signed=signed
        )

    def track_records(self):
        """Return FIT record locations with time, position, and speed values."""
        result = []
        for message in self.messages:
            if message.global_id != 20:
                continue
            fields = message.fields
            if not all(field_id in fields for field_id in (253, 0, 1)):
                continue
            timestamp = self.scalar(fields[253]) + FIT_EPOCH_UNIX
            latitude = self.scalar(fields[0], signed=True) * 180.0 / 2**31
            longitude = self.scalar(fields[1], signed=True) * 180.0 / 2**31
            speed = None
            if 73 in fields:
                raw = self.scalar(fields[73])
                if raw != 0xFFFFFFFF:
                    speed = raw / 1000.0
            elif 6 in fields:
                raw = self.scalar(fields[6])
                if raw != 0xFFFF:
                    speed = raw / 1000.0
            result.append(TrackPoint(
                message, timestamp * 1000, latitude, longitude, speed
            ))
        return result

    def gps_metadata_points(self):
        """Infer times for Garmin's untimestamped ~1 Hz GPS metadata stream."""
        groups = []
        previous_time = None
        first_time = None
        pending = []
        leading = []
        for message in self.messages:
            if message.global_id == 160 and 4 in message.fields:
                (pending if previous_time is not None else leading).append(message)
                continue
            if message.global_id != 20 or 253 not in message.fields:
                continue
            current_time = self.scalar(message.fields[253]) + FIT_EPOCH_UNIX
            if first_time is None:
                first_time = current_time
            if previous_time is not None and pending:
                spacing = (current_time - previous_time) / len(pending)
                groups.extend(
                    (item, previous_time + index * spacing)
                    for index, item in enumerate(pending)
                )
            previous_time = current_time
            pending = []
        if leading and first_time is not None:
            groups[:0] = [
                (item, first_time - len(leading) + index)
                for index, item in enumerate(leading)
            ]
        if pending and previous_time is not None:
            groups.extend(
                (item, previous_time + index)
                for index, item in enumerate(pending)
            )
        result = []
        for message, timestamp in groups:
            raw = self.scalar(message.fields[4])
            if raw != 0xFFFFFFFF:
                result.append((message, float(timestamp), raw / 1000.0))
        return result

    def set_record_speed(self, message, speed):
        millimetres_per_second = max(0, round(float(speed) * 1000))
        if 73 in message.fields:
            self.set_scalar(message.fields[73], min(millimetres_per_second, 0xFFFFFFFE))
        if 6 in message.fields:
            self.set_scalar(message.fields[6], min(millimetres_per_second, 0xFFFE))

    def set_gps_metadata_speed(self, message, speed):
        value = max(0, min(round(float(speed) * 1000), 0xFFFFFFFE))
        self.set_scalar(message.fields[4], value)

    def shift_timestamps(self, seconds):
        """Shift standard FIT date_time fields while preserving every message."""
        seconds = round(seconds)
        extra_date_fields = {
            0: (4,),       # file_id.time_created
            18: (2,),      # session.start_time
            19: (2,),      # lap.start_time
            34: (5,),      # activity.local_timestamp
        }
        for message in self.messages:
            fields = message.fields
            ids = ([253] if 253 in fields else []) + list(extra_date_fields.get(message.global_id, ()))
            for field_id in ids:
                field = fields.get(field_id)
                if field is None or field.size != 4:
                    continue
                value = self.scalar(field)
                if value not in (0, 0xFFFFFFFF):
                    self.set_scalar(field, value + seconds)

    def write(self, output):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.header_size >= 14:
            struct.pack_into("<H", self.data, 12, crc16(self.data[:12]))
        final_crc = crc16(self.data[:self.records_end])
        struct.pack_into("<H", self.data, self.records_end, final_crc)
        output.write_bytes(self.data)
