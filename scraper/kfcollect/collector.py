"""The always-on collection loop."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import signal
import sqlite3
import threading
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

from .config import Config
from .http import Oba, fetch
from .store import (
    STATUS_INSERT, RawArchive, compress_json, connect, free_gb, log_event, now_ms, status_row,
)

log = logging.getLogger("kfcollect")

GTFS_RT_FEEDS = ("vehicle-positions", "trip-updates", "service-alerts")


class Collector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db: sqlite3.Connection = connect(cfg.db_path)
        self.raw = RawArchive(cfg.raw_dir, cfg.timezone)
        self.oba = Oba(cfg.oba_base_url, cfg.oba_flavor, cfg.oba_api_key)
        self.stop = False
        self.cycle = 0
        # "Once per X" bookkeeping, rebuilt from the DB on start so restarts don't duplicate.
        self.seen_schedules: set[tuple[str, int]] = set()
        self.seen_refs: set[tuple[str, str, str]] = set()  # (day, table, id)
        self.seen_shapes: set[str] = {r[0] for r in self.db.execute("SELECT shape_id FROM shapes")}
        self.campus_stops: list[str] = []
        self.campus_routes: set[str] = set(cfg.focus_routes)  # routes serving the campus stops
        self.campus_active = 0     # tracked buses on those routes, last cycle
        self.trips_listed = 0      # trips (any state) listed across all routes, last cycle
        self.idle_cycles = 0       # consecutive cycles with no trips listed anywhere
        self.wake = threading.Event()
        self.stop_cursor = 0
        self.active_vehicles = 0
        self.minute_counts = {"ok": 0, "failed": 0}
        self.current_minute = int(time.time() // 60)
        self.consecutive_failures = 0
        self.last_ok_poll = 0
        self.last_daily_day = ""
        self.last_disk_check = 0.0
        self._load_seen()

    # --- bookkeeping ---------------------------------------------------------------

    def _today(self) -> str:
        return datetime.now(self.cfg.timezone).strftime("%Y-%m-%d")

    def _load_seen(self) -> None:
        since = int((time.time() - 3 * 86400) * 1000)
        for r in self.db.execute("SELECT trip_id, service_date FROM trip_schedules WHERE service_date >= ?", (since,)):
            self.seen_schedules.add((r[0], r[1]))
        today = self._today()
        for table, col in (("ref_trips", "trip_id"), ("ref_stops", "stop_id"), ("ref_routes", "route_id")):
            for r in self.db.execute(f"SELECT {col} FROM {table} WHERE day = ?", (today,)):
                self.seen_refs.add((today, table, r[0]))
        self._refresh_campus_stops()

    def _bootstrap_stops(self) -> None:
        """Load the focus routes' stops from stops-for-route, which answers even when no trips run."""
        for route in self.cfg.focus_routes:
            requested_at = now_ms()
            resp = self.oba.get("stops-for-route", route)
            self.raw.write("stops_for_route", route, resp.status, resp.body)
            self._record_poll("stops_for_route", route, requested_at, resp)
            if resp.ok:
                try:
                    refs = (resp.json().get("data") or {}).get("references") or {}
                    self._store_refs({"stops": refs.get("stops"), "routes": refs.get("routes")}, requested_at)
                except Exception:
                    pass
        self.db.commit()
        self._refresh_campus_stops()

    def _refresh_campus_stops(self) -> None:
        """Stops served by any focus route, from the latest reference snapshot."""
        focus = set(self.cfg.focus_routes)
        rows = self.db.execute(
            "SELECT stop_id, route_ids FROM ref_stops WHERE day = (SELECT MAX(day) FROM ref_stops)"
        ).fetchall()
        served = {r[0]: set(json.loads(r[1] or "[]")) for r in rows}
        stops = sorted(sid for sid, routes in served.items() if focus & routes)
        if stops:
            self.campus_stops = stops
            self.campus_routes = focus.union(*(served[sid] for sid in stops))

    def _record_poll(self, kind, target, requested_at, resp, server_time=None, n_rows=None, n_active=None) -> int:
        cur = self.db.execute(
            "INSERT INTO polls (cycle, kind, target, requested_at, server_time, http_status, elapsed_ms, bytes, "
            "n_rows, n_active, error) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (self.cycle, kind, target, requested_at, server_time, resp.status, resp.elapsed_ms, len(resp.body),
             n_rows, n_active, resp.error or None),
        )
        self.minute_counts["ok" if resp.ok and not resp.error else "failed"] += 1
        return cur.lastrowid

    # --- trips-for-route ---------------------------------------------------------------

    def poll_routes(self) -> None:
        self.cycle += 1
        active_total = campus_active = listed = 0
        any_ok = False
        for route in self.cfg.routes:
            requested_at = now_ms()
            resp = self.oba.get("trips-for-route", route, includeStatus=True, includeSchedule=True)
            if not resp.ok:
                self.raw.write("tfr", route, resp.status, resp.body)  # keep error bodies
                self._record_poll("tfr", route, requested_at, resp)
                continue
            try:
                payload = resp.json()
                if payload.get("code") not in (200, None):
                    raise ValueError(f"OBA code {payload.get('code')}: {payload.get('text')}")
                data = payload.get("data") or {}
                # An empty answer ("no trips, no alerts") carries nothing to re-parse later:
                # the polls row below records that we asked, so skip the raw copy.
                if data.get("list") or (data.get("references") or {}).get("situations"):
                    self.raw.write("tfr", route, resp.status, resp.body)
                n_rows, n_active = self._ingest_tfr(route, requested_at, resp, payload)
                active_total += n_active
                listed += n_rows
                if route in self.campus_routes:
                    campus_active += n_active
                any_ok = True
            except Exception as e:  # malformed payload: keep the raw copy, log the error
                self.raw.write("tfr", route, resp.status, resp.body)
                resp.error = f"parse: {type(e).__name__}: {e}"
                self._record_poll("tfr", route, requested_at, resp)
        self.db.commit()
        self.active_vehicles = active_total
        self.campus_active = campus_active
        self.trips_listed = listed
        self.idle_cycles = self.idle_cycles + 1 if any_ok and listed == 0 else 0
        if any_ok:
            self.last_ok_poll = now_ms()
            if self.consecutive_failures >= 3:
                log_event(self.db, "info", "feed_recovered", f"Feed recovered after {self.consecutive_failures} failed cycles")
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            if self.consecutive_failures in (3, 20, 120):
                log_event(self.db, "error", "feed_down", f"{self.consecutive_failures} consecutive poll cycles failed")

    def _ingest_tfr(self, route, requested_at, resp, payload) -> tuple[int, int]:
        server_time = int(payload.get("currentTime") or requested_at)
        data = payload.get("data") or {}
        trips = data.get("list") or []
        refs = data.get("references") or {}
        n_active = sum(1 for t in trips if (t.get("status") or {}).get("vehicleId")
                       and (t.get("status") or {}).get("predicted"))
        poll_id = self._record_poll("tfr", route, requested_at, resp, server_time, len(trips), n_active)
        rows = [status_row(poll_id, server_time, route, t) for t in trips if t.get("tripId")]
        if rows:
            self.db.executemany(STATUS_INSERT, rows)
        for t in trips:
            self._store_schedule(route, t, server_time)
        self._store_refs(refs, server_time)
        return len(rows), n_active

    def _store_schedule(self, route, trip, seen_at) -> None:
        sched = trip.get("schedule")
        tid = trip.get("tripId")
        sdate = (trip.get("status") or {}).get("serviceDate") or trip.get("serviceDate")
        if not sched or not tid or not sdate or (tid, sdate) in self.seen_schedules:
            return
        self.seen_schedules.add((tid, sdate))
        self.db.execute(
            "INSERT OR IGNORE INTO trip_schedules VALUES (?,?,?,?,?,?,?,?)",
            (tid, sdate, route, sched.get("timeZone"), sched.get("previousTripId") or None,
             sched.get("nextTripId") or None, json.dumps(sched.get("frequency")) if sched.get("frequency") else None,
             seen_at),
        )
        self.db.executemany(
            "INSERT OR IGNORE INTO stop_times VALUES (?,?,?,?,?,?,?,?,?)",
            [(tid, sdate, i, st.get("stopId"), st.get("arrivalTime"), st.get("departureTime"),
              st.get("distanceAlongTrip"), st.get("historicalOccupancy") or None, st.get("stopHeadsign") or None)
             for i, st in enumerate(sched.get("stopTimes") or [])],
        )

    def _store_refs(self, refs: dict, seen_at: int) -> None:
        day = self._today()
        new_stops = False
        for t in refs.get("trips") or []:
            if (day, "ref_trips", t["id"]) in self.seen_refs:
                continue
            self.seen_refs.add((day, "ref_trips", t["id"]))
            self.db.execute(
                "INSERT OR IGNORE INTO ref_trips VALUES (?,?,?,?,?,?,?,?,?,?)",
                (day, t["id"], t.get("routeId"), t.get("blockId"), t.get("directionId"), t.get("shapeId"),
                 t.get("tripHeadsign"), t.get("serviceId"), t.get("tripShortName"), json.dumps(t)),
            )
            if t.get("shapeId") and t["shapeId"] not in self.seen_shapes:
                self._fetch_shape(t["shapeId"])
        for s in refs.get("stops") or []:
            if (day, "ref_stops", s["id"]) in self.seen_refs:
                continue
            self.seen_refs.add((day, "ref_stops", s["id"]))
            new_stops = True
            self.db.execute(
                "INSERT OR IGNORE INTO ref_stops VALUES (?,?,?,?,?,?,?,?,?)",
                (day, s["id"], s.get("name"), s.get("lat"), s.get("lon"), s.get("direction"), s.get("code"),
                 json.dumps(s.get("routeIds") or []), json.dumps(s)),
            )
        for r in refs.get("routes") or []:
            if (day, "ref_routes", r["id"]) in self.seen_refs:
                continue
            self.seen_refs.add((day, "ref_routes", r["id"]))
            self.db.execute(
                "INSERT OR IGNORE INTO ref_routes VALUES (?,?,?,?,?,?,?)",
                (day, r["id"], r.get("agencyId"), r.get("shortName"), r.get("longName"), r.get("color"), json.dumps(r)),
            )
        for sit in refs.get("situations") or []:
            sid = sit.get("id")
            if not sid:
                continue
            cur = self.db.execute("UPDATE situations SET last_seen = ? WHERE situation_id = ?", (seen_at, sid))
            if cur.rowcount == 0:
                self.db.execute("INSERT INTO situations VALUES (?,?,?,?)", (sid, seen_at, seen_at, json.dumps(sit)))
                summary = (sit.get("summary") or {}).get("value") or sid
                log_event(self.db, "info", "alert", f"New service alert: {summary[:180]}")
        if new_stops:
            self._refresh_campus_stops()

    def _fetch_shape(self, shape_id: str) -> None:
        self.seen_shapes.add(shape_id)
        requested_at = now_ms()
        resp = self.oba.get("shape", shape_id)
        self.raw.write("shape", shape_id, resp.status, resp.body)
        self._record_poll("shape", shape_id, requested_at, resp)
        if resp.ok:
            try:
                pts = (resp.json().get("data") or {}).get("entry", {}).get("points")
                self.db.execute("INSERT OR REPLACE INTO shapes VALUES (?,?,?)", (shape_id, pts, requested_at))
            except Exception:
                self.seen_shapes.discard(shape_id)
        else:
            self.seen_shapes.discard(shape_id)

    # --- arrivals-and-departures-for-stop (round robin) ----------------------------------

    def poll_next_stop(self) -> None:
        if not self.campus_stops:
            return
        stop_id = self.campus_stops[self.stop_cursor % len(self.campus_stops)]
        self.stop_cursor += 1
        requested_at = now_ms()
        resp = self.oba.get("arrivals-and-departures-for-stop", stop_id, minutesBefore=5, minutesAfter=60)
        self.raw.write("arrivals", stop_id, resp.status, resp.body)
        if not resp.ok:
            self._record_poll("arrivals", stop_id, requested_at, resp)
            return
        try:
            payload = resp.json()
            server_time = payload.get("currentTime")
            ads = ((payload.get("data") or {}).get("entry") or {}).get("arrivalsAndDepartures") or []
            poll_id = self._record_poll("arrivals", stop_id, requested_at, resp, server_time, len(ads))
            keep = {"routeId", "tripId", "serviceDate", "vehicleId", "predicted", "predictedArrivalTime",
                    "scheduledArrivalTime", "predictedDepartureTime", "scheduledDepartureTime", "numberOfStopsAway",
                    "distanceFromStop", "status"}
            self.db.executemany(
                "INSERT INTO arrivals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(poll_id, server_time, stop_id, a.get("routeId"), a.get("tripId"), a.get("serviceDate"),
                  a.get("vehicleId") or None, 1 if a.get("predicted") else 0, a.get("predictedArrivalTime") or None,
                  a.get("scheduledArrivalTime"), a.get("predictedDepartureTime") or None,
                  a.get("scheduledDepartureTime"), a.get("numberOfStopsAway"), a.get("distanceFromStop"),
                  a.get("status"), json.dumps({k: v for k, v in a.items() if k not in keep and k != "tripStatus"}))
                 for a in ads],
            )
            self.db.commit()
        except Exception as e:
            resp.error = f"parse: {type(e).__name__}: {e}"
            self._record_poll("arrivals", stop_id, requested_at, resp)

    # --- GTFS-Realtime (Spare), stored raw ------------------------------------------------

    def poll_gtfs_rt(self) -> None:
        for feed in GTFS_RT_FEEDS:
            requested_at = now_ms()
            resp = fetch(f"{self.cfg.gtfs_base}/gtfs-rt/{feed}.pb", retries=0)
            # A feed with no entities is just a ~15-byte header (version + timestamp); don't archive it.
            if not resp.ok or len(resp.body) > 64:
                self.raw.write("gtfs_rt", feed, resp.status, resp.body, binary=True)
            self._record_poll("gtfs_rt", feed, requested_at, resp, n_rows=None)
        self.db.commit()

    # --- daily snapshots ------------------------------------------------------------------

    def daily_snapshots(self) -> None:
        day = self._today()
        if day == self.last_daily_day or not self.campus_stops:
            return
        self.last_daily_day = day
        self._snapshot_gtfs_static()
        self._snapshot_timetables(day)
        self.db.commit()

    def _snapshot_gtfs_static(self) -> None:
        requested_at = now_ms()
        resp = fetch(f"{self.cfg.gtfs_base}/gtfs.zip", timeout=60)
        self._record_poll("gtfs_static", "gtfs.zip", requested_at, resp)
        if not resp.ok:
            return
        sha = hashlib.sha256(resp.body).hexdigest()
        if self.db.execute("SELECT 1 FROM gtfs_versions WHERE sha256 = ?", (sha,)).fetchone():
            return
        feed_version = ""
        try:
            with zipfile.ZipFile(io.BytesIO(resp.body)) as z:
                info = z.read("feed_info.txt").decode("utf-8-sig").splitlines()
                header, values = info[0].split(","), info[1].split(",")
                feed_version = dict(zip(header, values)).get("feed_version", "")
        except Exception:
            pass
        self.cfg.gtfs_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.gtfs_dir / f"gtfs-{datetime.now(self.cfg.timezone):%Y%m%d}-{sha[:10]}.zip"
        path.write_bytes(resp.body)
        self.db.execute("INSERT INTO gtfs_versions VALUES (?,?,?,?,?)",
                        (sha, requested_at, feed_version, str(path), len(resp.body)))
        log_event(self.db, "info", "gtfs_static", f"New GTFS static version saved (feed_version {feed_version or '?'})")

    def _snapshot_timetables(self, day: str) -> None:
        """schedule-for-stop for every campus stop: the timetable as published to riders that day."""
        for stop_id in self.campus_stops:
            if self.db.execute("SELECT 1 FROM timetables WHERE day = ? AND stop_id = ?", (day, stop_id)).fetchone():
                continue
            requested_at = now_ms()
            resp = self.oba.get("schedule-for-stop", stop_id, date=day)
            self._record_poll("timetable", stop_id, requested_at, resp)
            if resp.ok:
                try:
                    self.db.execute("INSERT OR REPLACE INTO timetables VALUES (?,?,?,?)",
                                    (day, stop_id, requested_at, compress_json(resp.json())))
                except Exception:
                    pass
            time.sleep(0.5)

    # --- health -----------------------------------------------------------------------------

    def heartbeat(self) -> None:
        minute = int(time.time() // 60)
        if minute != self.current_minute:
            self.db.execute(
                "INSERT OR REPLACE INTO heartbeat VALUES (?,?,?,?)",
                (self.current_minute, self.minute_counts["ok"], self.minute_counts["failed"], self.active_vehicles),
            )
            self.db.commit()
            self.minute_counts = {"ok": 0, "failed": 0}
            self.current_minute = minute
        hb = {
            "ts": now_ms(), "last_ok_poll": self.last_ok_poll, "cycle": self.cycle,
            "active_vehicles": self.active_vehicles, "consecutive_failures": self.consecutive_failures,
            "campus_stops": len(self.campus_stops),
        }
        tmp = self.cfg.heartbeat_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(hb))
        tmp.replace(self.cfg.heartbeat_path)

    def disk_guard(self) -> None:
        if time.time() - self.last_disk_check < 600:
            return
        self.last_disk_check = time.time()
        days = sorted(p for p in self.cfg.raw_dir.glob("????-??-??") if p.is_dir())
        if self.cfg.raw_keep_days > 0:
            cutoff = (datetime.now(self.cfg.timezone) - timedelta(days=self.cfg.raw_keep_days)).strftime("%Y-%m-%d")
            for d in days:
                if d.name < cutoff:
                    _rmtree(d)
        # Out of space: drop the oldest raw days (never the last 3); parsed tables are kept.
        days = sorted(p for p in self.cfg.raw_dir.glob("????-??-??") if p.is_dir())
        while free_gb(self.cfg.data_dir) < self.cfg.min_free_gb and len(days) > 3:
            victim = days.pop(0)
            _rmtree(victim)
            log_event(self.db, "warning", "disk", f"Low disk: deleted raw archive {victim.name}")

    # --- main loop ---------------------------------------------------------------------------

    def run(self) -> None:
        def _stop(*_):
            self.stop = True
            self.wake.set()  # end the current sleep right away

        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
        log_event(self.db, "info", "start", f"Collector started: {len(self.cfg.routes)} routes, "
                                            f"poll {self.cfg.poll_interval_s}s")
        log.info("routes: %s", ", ".join(self.cfg.routes))
        self._bootstrap_stops()
        inf = float("inf")
        next_routes = time.monotonic()
        next_rt = next_routes if self.cfg.gtfs_rt_interval_s > 0 else inf
        next_stop = next_routes if self.cfg.arrivals_interval_s > 0 else inf
        while not self.stop:
            now = time.monotonic()
            try:
                # After 2 minutes with no trips listed on any route (e.g. 3 a.m.), poll every
                # KF_IDLE_POLL_INTERVAL_S instead; the first listed trip switches back to full speed.
                idle = self.idle_cycles >= 4
                if now >= next_routes:
                    self.poll_routes()
                    idle = self.idle_cycles >= 4  # poll_routes just updated it
                    next_routes = now + (self.cfg.idle_poll_interval_s if idle else self.cfg.poll_interval_s)
                    self.daily_snapshots()
                if now >= next_rt:
                    next_rt = now + (self.cfg.idle_poll_interval_s if idle else self.cfg.gtfs_rt_interval_s)
                    self.poll_gtfs_rt()
                if now >= next_stop:
                    # One pass over the campus stops per interval, only while buses run on routes
                    # that serve them (a late trolley elsewhere doesn't count).
                    per_stop = self.cfg.arrivals_interval_s / max(1, len(self.campus_stops))
                    next_stop = now + (per_stop if self.campus_active else 60)
                    if self.campus_active:
                        self.poll_next_stop()
                self.heartbeat()
                self.raw.tick()
                self.disk_guard()
            except sqlite3.OperationalError as e:
                log.exception("database error")
                self.wake.wait(5)
                try:
                    log_event(self.db, "error", "db", str(e))
                except Exception:
                    pass
            except Exception as e:  # never die on one bad cycle
                log.exception("cycle error")
                try:
                    log_event(self.db, "error", "cycle", f"{type(e).__name__}: {e}")
                except Exception:
                    pass
                self.wake.wait(2)
            self.wake.wait(max(0.2, min(next_routes, next_rt, next_stop) - time.monotonic()))
        self.raw.close()
        log_event(self.db, "info", "stop", "Collector stopped")
        self.db.close()


def _rmtree(path: Path) -> None:
    for p in sorted(path.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    path.rmdir()
