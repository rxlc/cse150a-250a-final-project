"""SQLite schema and writers, plus the hourly gzip raw archive."""

from __future__ import annotations

import base64
import gzip
import json
import os
import shutil
import sqlite3
import time
import zlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- One row per HTTP request the collector makes (any endpoint).
CREATE TABLE IF NOT EXISTS polls (
  id INTEGER PRIMARY KEY,
  cycle INTEGER,                 -- trips-for-route requests in the same 30 s cycle share a cycle
  kind TEXT NOT NULL,            -- tfr | arrivals | gtfs_rt | timetable | gtfs_static | shape
  target TEXT,                   -- route id, stop id, feed name...
  requested_at INTEGER NOT NULL, -- local wall clock, epoch ms
  server_time INTEGER,           -- OBA currentTime (ms), when present
  http_status INTEGER,
  elapsed_ms INTEGER,
  bytes INTEGER,
  n_rows INTEGER,
  n_active INTEGER,
  error TEXT
);
CREATE INDEX IF NOT EXISTS polls_time ON polls(requested_at);
CREATE INDEX IF NOT EXISTS polls_kind_time ON polls(kind, requested_at);

-- trips-for-route: one row per trip per poll, every status field (HMM proposal, "Fields collected").
CREATE TABLE IF NOT EXISTS trip_status (
  poll_id INTEGER NOT NULL,
  server_time INTEGER NOT NULL,
  route_id TEXT NOT NULL,
  trip_id TEXT NOT NULL,
  active_trip_id TEXT,
  service_date INTEGER,
  vehicle_id TEXT,
  block_trip_sequence INTEGER,
  predicted INTEGER,
  phase TEXT,
  status TEXT,
  schedule_deviation REAL,
  pos_lat REAL, pos_lon REAL,
  gps_lat REAL, gps_lon REAL,
  orientation REAL,
  last_known_orientation REAL,
  distance_along_trip REAL,
  scheduled_distance_along_trip REAL,
  total_distance_along_trip REAL,
  last_known_distance_along_trip REAL,
  closest_stop TEXT,
  closest_stop_time_offset INTEGER,
  next_stop TEXT,
  next_stop_time_offset INTEGER,
  last_location_update_time INTEGER,
  last_update_time INTEGER,
  occupancy_status TEXT,
  occupancy_count INTEGER,
  occupancy_capacity INTEGER,
  situation_ids TEXT,            -- JSON array
  frequency TEXT,                -- JSON, set for headway-based trips
  vehicle_features TEXT,         -- JSON array
  extra TEXT                     -- JSON of any status fields not listed above (future-proofing)
);
CREATE INDEX IF NOT EXISTS ts_trip ON trip_status(service_date, trip_id, server_time);
CREATE INDEX IF NOT EXISTS ts_time ON trip_status(server_time);
CREATE INDEX IF NOT EXISTS ts_route_time ON trip_status(route_id, server_time);

-- Each trip's schedule, once per trip per service day.
CREATE TABLE IF NOT EXISTS trip_schedules (
  trip_id TEXT NOT NULL,
  service_date INTEGER NOT NULL,
  route_id TEXT,
  time_zone TEXT,
  previous_trip_id TEXT,
  next_trip_id TEXT,
  frequency TEXT,
  first_seen INTEGER,
  PRIMARY KEY (trip_id, service_date)
);
CREATE TABLE IF NOT EXISTS stop_times (
  trip_id TEXT NOT NULL,
  service_date INTEGER NOT NULL,
  seq INTEGER NOT NULL,
  stop_id TEXT,
  arrival_time INTEGER,          -- seconds after service_date
  departure_time INTEGER,
  distance_along_trip REAL,
  historical_occupancy TEXT,
  stop_headsign TEXT,
  PRIMARY KEY (trip_id, service_date, seq)
);

-- Reference tables, once per local day (they change rarely).
CREATE TABLE IF NOT EXISTS ref_trips (
  day TEXT NOT NULL, trip_id TEXT NOT NULL, route_id TEXT, block_id TEXT, direction_id TEXT,
  shape_id TEXT, headsign TEXT, service_id TEXT, short_name TEXT, json TEXT,
  PRIMARY KEY (day, trip_id)
);
CREATE TABLE IF NOT EXISTS ref_stops (
  day TEXT NOT NULL, stop_id TEXT NOT NULL, name TEXT, lat REAL, lon REAL, direction TEXT,
  code TEXT, route_ids TEXT, json TEXT,
  PRIMARY KEY (day, stop_id)
);
CREATE TABLE IF NOT EXISTS ref_routes (
  day TEXT NOT NULL, route_id TEXT NOT NULL, agency_id TEXT, short_name TEXT, long_name TEXT,
  color TEXT, json TEXT,
  PRIMARY KEY (day, route_id)
);
-- Service alerts, whenever a situation id appears; last_seen updates while it stays active.
CREATE TABLE IF NOT EXISTS situations (
  situation_id TEXT PRIMARY KEY, first_seen INTEGER, last_seen INTEGER, json TEXT
);
CREATE TABLE IF NOT EXISTS shapes (
  shape_id TEXT PRIMARY KEY, points TEXT, fetched_at INTEGER
);

-- arrivals-and-departures-for-stop: the predictions riders actually saw (a calibration baseline).
CREATE TABLE IF NOT EXISTS arrivals (
  poll_id INTEGER NOT NULL,
  server_time INTEGER,
  stop_id TEXT NOT NULL,
  route_id TEXT,
  trip_id TEXT,
  service_date INTEGER,
  vehicle_id TEXT,
  predicted INTEGER,
  predicted_arrival INTEGER,
  scheduled_arrival INTEGER,
  predicted_departure INTEGER,
  scheduled_departure INTEGER,
  stops_away INTEGER,
  distance_from_stop REAL,
  status TEXT,
  extra TEXT
);
CREATE INDEX IF NOT EXISTS arr_stop_time ON arrivals(stop_id, server_time);

-- Published timetables: schedule-for-stop per stop per day (compressed JSON), and GTFS static versions.
CREATE TABLE IF NOT EXISTS timetables (
  day TEXT NOT NULL, stop_id TEXT NOT NULL, fetched_at INTEGER, json_z BLOB,
  PRIMARY KEY (day, stop_id)
);
CREATE TABLE IF NOT EXISTS gtfs_versions (
  sha256 TEXT PRIMARY KEY, fetched_at INTEGER, feed_version TEXT, path TEXT, bytes INTEGER
);

-- Collector health.
CREATE TABLE IF NOT EXISTS heartbeat (
  minute INTEGER PRIMARY KEY,     -- epoch minute (local wall clock)
  polls_ok INTEGER, polls_failed INTEGER, active_vehicles INTEGER
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY, ts INTEGER, level TEXT, kind TEXT, message TEXT
);
"""

STATUS_COLUMNS = {
    "activeTripId", "serviceDate", "vehicleId", "blockTripSequence", "predicted", "phase", "status",
    "scheduleDeviation", "position", "lastKnownLocation", "orientation", "lastKnownOrientation",
    "distanceAlongTrip", "scheduledDistanceAlongTrip", "totalDistanceAlongTrip", "lastKnownDistanceAlongTrip",
    "closestStop", "closestStopTimeOffset", "nextStop", "nextStopTimeOffset", "lastLocationUpdateTime",
    "lastUpdateTime", "occupancyStatus", "occupancyCount", "occupancyCapacity", "situationIds", "frequency",
    "vehicleFeatures",
}


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def now_ms() -> int:
    return int(time.time() * 1000)


def log_event(db: sqlite3.Connection, level: str, kind: str, message: str) -> None:
    db.execute("INSERT INTO events (ts, level, kind, message) VALUES (?,?,?,?)", (now_ms(), level, kind, message))
    db.commit()


def _ll(obj, key):
    v = (obj or {}).get(key) if isinstance(obj, dict) else None
    return (v.get("lat"), v.get("lon")) if isinstance(v, dict) else (None, None)


def _nonneg(v):
    return v if isinstance(v, (int, float)) and v >= 0 else None


def status_row(poll_id: int, server_time: int, route_id: str, trip: dict) -> tuple:
    s = trip.get("status") or {}
    pos = _ll(s, "position")
    gps = _ll(s, "lastKnownLocation")
    extra = {k: v for k, v in s.items() if k not in STATUS_COLUMNS}
    j = lambda v: json.dumps(v) if v not in (None, [], {}) else None  # noqa: E731
    return (
        poll_id, server_time, route_id, trip.get("tripId"), s.get("activeTripId"),
        s.get("serviceDate", trip.get("serviceDate")), s.get("vehicleId") or None, s.get("blockTripSequence"),
        1 if s.get("predicted") else 0, s.get("phase") or None, s.get("status") or None,
        s.get("scheduleDeviation"), pos[0], pos[1], gps[0], gps[1], s.get("orientation"),
        s.get("lastKnownOrientation"), s.get("distanceAlongTrip"), s.get("scheduledDistanceAlongTrip"),
        s.get("totalDistanceAlongTrip"), s.get("lastKnownDistanceAlongTrip"), s.get("closestStop") or None,
        s.get("closestStopTimeOffset"), s.get("nextStop") or None, s.get("nextStopTimeOffset"),
        s.get("lastLocationUpdateTime") or None, s.get("lastUpdateTime") or None,
        s.get("occupancyStatus") or None,
        _nonneg(s.get("occupancyCount")), _nonneg(s.get("occupancyCapacity")),  # OBA uses -1 for "unknown"
        j(s.get("situationIds") or trip.get("situationIds")), j(s.get("frequency") or trip.get("frequency")),
        j(s.get("vehicleFeatures")), j(extra),
    )


STATUS_INSERT = "INSERT INTO trip_status VALUES (" + ",".join("?" * 35) + ")"


class RawArchive:
    """Every response, in data/raw/YYYY-MM-DD/HH.jsonl.gz (local time).

    Records are buffered and appended as a complete gzip member at most every
    `flush_s` seconds (and on hour change / shutdown). Concatenated gzip members are a
    valid .gz file, so every file is always readable with zcat or gzip.open, even after
    a crash; at most `flush_s` seconds of raw records are lost (the parsed tables still
    have them).
    """

    def __init__(self, root: Path, tz: ZoneInfo, flush_s: float = 60.0, max_buffer: int = 4_000_000):
        self.root = root
        self.tz = tz
        self.flush_s = flush_s
        self.max_buffer = max_buffer
        self._key: str | None = None
        self._buf: list[bytes] = []
        self._size = 0
        self._last_flush = time.monotonic()

    def write(self, kind: str, target: str, status: int, body: bytes, binary: bool = False) -> None:
        key = datetime.now(self.tz).strftime("%Y-%m-%d/%H")
        if key != self._key:
            self.flush()
            self._key = key
        rec = {"t": now_ms(), "kind": kind, "target": target, "status": status}
        if binary:
            rec["b64"] = base64.b64encode(body).decode()
        else:
            rec["body"] = body.decode("utf-8", "replace")
        line = (json.dumps(rec) + "\n").encode()
        self._buf.append(line)
        self._size += len(line)
        if self._size >= self.max_buffer or time.monotonic() - self._last_flush >= self.flush_s:
            self.flush()

    def flush(self) -> None:
        self._last_flush = time.monotonic()
        if not self._buf or not self._key:
            return
        path = self.root / f"{self._key}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        member = gzip.compress(b"".join(self._buf), compresslevel=6)
        with open(path, "ab") as fh:
            fh.write(member)
            fh.flush()
            os.fsync(fh.fileno())
        self._buf = []
        self._size = 0

    def tick(self) -> None:
        """Called every loop: writes out buffered records once they're `flush_s` old, even if no new ones arrive."""
        if self._buf and time.monotonic() - self._last_flush >= self.flush_s:
            self.flush()

    def close(self) -> None:
        self.flush()


def compress_json(obj) -> bytes:
    return zlib.compress(json.dumps(obj).encode(), 6)


def free_gb(path: Path) -> float:
    return shutil.disk_usage(str(path)).free / 1e9


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0
