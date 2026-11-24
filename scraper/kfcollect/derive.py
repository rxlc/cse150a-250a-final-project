"""Turns raw polls into analysis tables (re-runnable; raw data is never modified).

Outputs, per service day:
  stop_passages   when each trip passed each scheduled stop, and its delay (submitted proposal)
  segments        delay gained between consecutive stops (HMM proposal, M2 observations)
  intervals_5min  per route/direction 5-minute slots: how long a rider waits for the next bus vs the
                  timetable (excess wait, the HMM observation) plus each trip's own delay (submitted proposal)
  vehicle_steps   per-bus 30 s steps: speed bin, at-stop and layover-zone flags (HMM proposal, M1)
  trip_events     bus swaps within a trip or block
  gaps            minutes where the collector got no successful poll

Progress along a trip is measured from the bus's own GPS fix (lastKnownLocation at
lastLocationUpdateTime) projected onto the trip's shape. OBA's distanceAlongTrip is kept
in trip_status for comparison, but on this feed it has been observed to equal the
*scheduled* distance, which would make every derived delay zero.
"""

from __future__ import annotations

import math
import sqlite3
import statistics
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta

from .config import Config

DERIVED_SCHEMA = """
CREATE TABLE IF NOT EXISTS stop_passages (
  service_date INTEGER NOT NULL, trip_id TEXT NOT NULL, seq INTEGER NOT NULL,
  route_id TEXT, direction_id TEXT, block_id TEXT, vehicle_id TEXT, stop_id TEXT,
  scheduled_time INTEGER,   -- epoch ms
  actual_time INTEGER,      -- epoch ms, interpolated between GPS fixes
  delay_s REAL,             -- actual - scheduled (+ = late)
  bracket_s REAL,           -- seconds between the two fixes used (data quality)
  obs_deviation_s REAL,     -- OBA scheduleDeviation at the nearest poll, for comparison
  PRIMARY KEY (service_date, trip_id, seq)
);
CREATE TABLE IF NOT EXISTS segments (
  service_date INTEGER NOT NULL, trip_id TEXT NOT NULL, seq INTEGER NOT NULL,
  route_id TEXT, direction_id TEXT, vehicle_id TEXT, from_stop TEXT, to_stop TEXT,
  t_from INTEGER, t_to INTEGER, delay_from REAL, delay_to REAL, gained_s REAL, gained_bin INTEGER,
  PRIMARY KEY (service_date, trip_id, seq)
);
CREATE TABLE IF NOT EXISTS intervals_5min (
  route_id TEXT NOT NULL, direction_id TEXT NOT NULL, slot_start INTEGER NOT NULL, slot_local TEXT,
  n INTEGER, mean_delay REAL, median_delay REAL, p90_delay REAL, max_delay REAL, mean_gained REAL,
  delay_level INTEGER,      -- from mean trip delay: 0 early, 1 on time, 2 late, 3 very late (KF_DEVIATION_LEVELS)
  n_stops INTEGER,          -- stops with scheduled service in this slot
  sched_wait_s REAL,        -- mean wait for the next bus, by the timetable, for a rider arriving at a random moment
  wait_s REAL,              -- the same, with the buses that actually came (any trip, any bus)
  excess_wait_s REAL,       -- wait_s - sched_wait_s
  max_excess_s REAL,        -- worst stop on the route
  missed_trips INTEGER,     -- scheduled trips starting in this slot that no bus was tracked on
  level INTEGER,            -- from excess wait: 0 shorter, 1 normal, 2 longer, 3 much longer (KF_WAIT_LEVELS)
  PRIMARY KEY (route_id, direction_id, slot_start)
);
CREATE TABLE IF NOT EXISTS vehicle_steps (
  vehicle_id TEXT NOT NULL, t INTEGER NOT NULL, service_date INTEGER, trip_id TEXT, route_id TEXT,
  along_m REAL, offset_m REAL, speed_mps REAL, speed_bin INTEGER, at_stop INTEGER, in_layover_zone INTEGER,
  PRIMARY KEY (vehicle_id, t)
);
CREATE TABLE IF NOT EXISTS trip_events (
  service_date INTEGER, trip_id TEXT, block_id TEXT, t INTEGER, kind TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS gaps (
  start_ms INTEGER PRIMARY KEY, end_ms INTEGER, minutes INTEGER
);
CREATE TABLE IF NOT EXISTS derive_runs (
  run_at INTEGER, service_date INTEGER, trips INTEGER, passages INTEGER, segments INTEGER, note TEXT
);
"""

STALE_MS = 120_000      # fixes older than this at poll time are not used (HMM proposal, step 3)
MAX_BRACKET_S = 240     # don't interpolate a stop passage across a longer feed gap
MAX_OFFSET_M = 80       # GPS fixes farther than this from the shape are ignored
EARLY_MS = 5 * 60 * 1000  # a bus picked up mid-trip is at most this far ahead of the timetable
AT_STOP_M = 30
LAYOVER_ZONE_M = 150
SPEED_BINS = (0.5, 4.0)  # m/s: stopped < 0.5 <= slow < 4 <= moving


# --- geometry ---------------------------------------------------------------------------

def decode_polyline(encoded: str) -> list[tuple[float, float]]:
    pts, idx, lat, lng = [], 0, 0, 0
    while idx < len(encoded):
        for axis in (0, 1):
            result, shift = 0, 0
            while True:
                b = ord(encoded[idx]) - 63
                idx += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if axis == 0:
                lat += delta
            else:
                lng += delta
        pts.append((lat / 1e5, lng / 1e5))
    return pts


def haversine(a, b) -> float:
    r = 6371008.8
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


class Shape:
    def __init__(self, pts: list[tuple[float, float]]):
        self.pts = pts
        self.cum = [0.0]
        for i in range(1, len(pts)):
            self.cum.append(self.cum[-1] + haversine(pts[i - 1], pts[i]))
        self.length = self.cum[-1] if self.cum else 0.0

    def project(self, p, lo: float = -math.inf, hi: float = math.inf) -> tuple[float, float]:
        """(distance along, offset) of the nearest point to p, searching only [lo, hi] along."""
        mlat = 111_320.0
        mlon = 111_320.0 * math.cos(math.radians(p[0]))
        best = (0.0, math.inf)
        for i in range(1, len(self.pts)):
            c0, c1 = self.cum[i - 1], self.cum[i]
            if c1 < lo or c0 > hi:
                continue
            ax, ay = (self.pts[i - 1][1] - p[1]) * mlon, (self.pts[i - 1][0] - p[0]) * mlat
            bx, by = (self.pts[i][1] - p[1]) * mlon, (self.pts[i][0] - p[0]) * mlat
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            seg = c1 - c0
            t = 0.0 if seg2 == 0 else -(ax * dx + ay * dy) / seg2
            tmin = max(0.0, (lo - c0) / seg) if seg else 0.0
            tmax = min(1.0, (hi - c0) / seg) if seg else 1.0
            t = min(max(t, tmin), tmax)
            off = math.hypot(ax + t * dx, ay + t * dy)
            if off < best[1]:
                best = (c0 + t * seg, off)
        return best


# --- helpers ------------------------------------------------------------------------------

def level_of(delay_s: float, bounds: list[int]) -> int:
    """With bounds (-60, 120, 300): 0 early, 1 on time, 2 late, 3 very late."""
    return bisect_right(bounds, delay_s)


def gained_bin(g: float, bins: list[int]) -> int:
    for i, b in enumerate(bins):
        if g <= b:
            return i
    return len(bins)


def speed_bin(v: float) -> int:
    return 0 if v < SPEED_BINS[0] else 1 if v < SPEED_BINS[1] else 2


def schedule_base(service_date: int, route_id: str, cfg: Config) -> tuple[int, float]:
    """Epoch ms that a trip's schedule seconds count from, and the matching fix for OBA's deviation.

    GTFS counts schedule seconds from "noon minus 12 h", which is local midnight except on
    daylight-saving days (01:00 on Nov 1, 23:00 the day before in March); OBA's serviceDate follows
    that. Some agencies' feeds (Triton Transit/Spare: KF_WALLCLOCK_AGENCIES) actually mean wall-clock
    time from midnight, so on DST days both OBA's scheduleDeviation and a naive serviceDate + seconds
    are off by an hour. For those agencies, count from local midnight and shift OBA's deviation to match.
    """
    agency = route_id.split("_", 1)[0] if route_id else ""
    if agency not in cfg.wallclock_agencies:
        return service_date, 0.0
    local = datetime.fromtimestamp(service_date / 1000, cfg.timezone)
    midnight = int(local.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    if local.hour >= 12:  # serviceDate at 23:00 the previous evening (DST start): midnight is the next day
        midnight = int((local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).timestamp() * 1000)
    return midnight, (service_date - midnight) / 1000


# --- per-day derivation ------------------------------------------------------------------------

def derive_day(db: sqlite3.Connection, cfg: Config, service_date: int) -> dict:
    shapes: dict[str, Shape] = {}
    for r in db.execute("SELECT shape_id, points FROM shapes WHERE points IS NOT NULL"):
        try:
            shapes[r["shape_id"]] = Shape(decode_polyline(r["points"]))
        except Exception:
            pass
    trip_meta = {}
    for r in db.execute("SELECT trip_id, route_id, block_id, direction_id, shape_id FROM ref_trips ORDER BY day"):
        trip_meta[r["trip_id"]] = dict(r)

    rows = db.execute(
        """SELECT trip_id, route_id, vehicle_id, server_time, schedule_deviation, gps_lat, gps_lon,
                  last_location_update_time AS llu, block_trip_sequence
           FROM trip_status
           WHERE service_date = ? AND predicted = 1 AND phase IS NOT NULL AND vehicle_id IS NOT NULL
             -- Rows for later trips in a block describe the bus on its *current* trip; skip them.
             AND (active_trip_id IS NULL OR active_trip_id = trip_id)
           ORDER BY trip_id, server_time""",
        (service_date,),
    ).fetchall()
    by_trip: dict[str, list] = defaultdict(list)
    for r in rows:
        by_trip[r["trip_id"]].append(r)

    for t in ("stop_passages", "segments", "trip_events"):
        db.execute(f"DELETE FROM {t} WHERE service_date = ?", (service_date,))
    db.execute("DELETE FROM vehicle_steps WHERE service_date = ?", (service_date,))

    n_pass = n_seg = 0
    for trip_id, obs in by_trip.items():
        meta = trip_meta.get(trip_id, {})
        shape = shapes.get(meta.get("shape_id") or "")
        stop_times = db.execute(
            "SELECT seq, stop_id, arrival_time, departure_time, distance_along_trip FROM stop_times "
            "WHERE trip_id = ? AND service_date = ? ORDER BY seq", (trip_id, service_date)).fetchall()
        if not shape or not stop_times:
            continue
        # The trip's own route. trips-for-route also lists other routes' trips in shared blocks
        # (Inside/Outside Loop, SuperLoop 201/202), so the route a row was polled under can be wrong.
        route_id = meta.get("route_id") or obs[0]["route_id"]
        sched_base, dev_shift = schedule_base(service_date, route_id, cfg)

        # 1. Deduplicate fixes and drop stale ones (one row per lastLocationUpdateTime).
        fixes, seen = [], set()
        for r in obs:
            if r["gps_lat"] is None or not r["llu"] or r["llu"] in seen:
                continue
            if r["server_time"] - r["llu"] > STALE_MS:
                continue
            seen.add(r["llu"])
            fixes.append(r)
        if len(fixes) < 2:
            continue

        # 2. Project fixes onto the shape, moving forward only (avoids snapping to the return leg).
        sched_t = [sched_base + int((st["departure_time"] if j == 0 else st["arrival_time"]) or 0) * 1000
                   for j, st in enumerate(stop_times)]
        sched_d = [st["distance_along_trip"] or 0.0 for st in stop_times]
        track = []  # (t_ms, along_m, offset_m, row)
        last_along = 0.0
        for r in fixes:
            p = (r["gps_lat"], r["gps_lon"])
            along, off = shape.project(p, last_along - 100, last_along + 3000)
            if not track and off > MAX_OFFSET_M:
                # Not near the start: the collector may have started mid-trip. But a bus can't be farther
                # along than the timetable allows (running up to EARLY_MS early): on loops, where the first
                # and last stops are the same place, a bus driving up to the start would otherwise match
                # the end of the trip, and every later fix would be lost behind it.
                k = bisect_right(sched_t, r["llu"] + EARLY_MS)
                limit = sched_d[min(k, len(sched_d) - 1)] if k else 0.0
                along, off = shape.project(p, -math.inf, limit + 100)
            if off > MAX_OFFSET_M:
                continue
            along = max(along, last_along)
            last_along = along
            track.append((r["llu"], along, off, r))
        if len(track) < 2:
            continue

        # Bus swaps within the trip.
        vehicles = [x[3]["vehicle_id"] for x in track]
        for i in range(1, len(vehicles)):
            if vehicles[i] != vehicles[i - 1]:
                db.execute("INSERT INTO trip_events VALUES (?,?,?,?,?,?)",
                           (service_date, trip_id, meta.get("block_id"), track[i][0], "bus_swap",
                            f"{vehicles[i - 1]} -> {vehicles[i]}"))

        # 3. Stop passages: first time the bus reaches each stop's distance along the trip.
        stop_along = [st["distance_along_trip"] or 0.0 for st in stop_times]
        passages = []
        for j, st in enumerate(stop_times):
            if j == 0:
                target = max(stop_along[j], 25.0)  # departure: the bus has pulled 25 m away from the first stop
            elif j == len(stop_times) - 1:
                target = stop_along[j] - 20.0  # buses end their trip just short of the terminal's exact point
            else:
                target = stop_along[j]
            for k in range(1, len(track)):
                (t0, d0, _, r0), (t1, d1, _, _) = track[k - 1], track[k]
                if d0 < target <= d1:
                    bracket = (t1 - t0) / 1000
                    if bracket > MAX_BRACKET_S:
                        break
                    t_pass = t0 + (target - d0) / (d1 - d0) * (t1 - t0)
                    sched_s = st["departure_time"] if j == 0 else st["arrival_time"]
                    sched = sched_base + int(sched_s) * 1000
                    nearest = min(obs, key=lambda r: abs(r["server_time"] - t_pass))
                    dev = nearest["schedule_deviation"]
                    passages.append((j, st, int(t_pass), sched, (t_pass - sched) / 1000, bracket,
                                     dev + dev_shift if dev is not None else None, r0["vehicle_id"]))
                    break
        db.executemany(
            "INSERT OR REPLACE INTO stop_passages VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(service_date, trip_id, j, route_id, meta.get("direction_id"), meta.get("block_id"), veh,
              st["stop_id"], sched, t_pass, delay, bracket, dev)
             for j, st, t_pass, sched, delay, bracket, dev, veh in passages],
        )
        n_pass += len(passages)

        # 4. Delay gained per segment between consecutive observed stops.
        segs = []
        for a, b in zip(passages, passages[1:]):
            g = b[4] - a[4]
            segs.append((service_date, trip_id, b[0], route_id, meta.get("direction_id"), b[7], a[1]["stop_id"],
                         b[1]["stop_id"], a[2], b[2], a[4], b[4], g, gained_bin(g, cfg.gained_bins)))
        db.executemany("INSERT OR REPLACE INTO segments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", segs)
        n_seg += len(segs)

        # 5. Vehicle steps for the behavior HMM.
        total = shape.length
        steps = []
        for k in range(1, len(track)):
            (t0, d0, _, _), (t1, d1, off, r) = track[k - 1], track[k]
            dt = (t1 - t0) / 1000
            if dt <= 0 or dt > MAX_BRACKET_S:
                continue
            v = (d1 - d0) / dt
            at_stop = any(abs(d1 - s) <= AT_STOP_M for s in stop_along)
            layover = d1 <= LAYOVER_ZONE_M or d1 >= total - LAYOVER_ZONE_M
            steps.append((r["vehicle_id"], t1, service_date, trip_id, route_id, d1, off, v, speed_bin(v),
                          int(at_stop), int(layover)))
        db.executemany("INSERT OR REPLACE INTO vehicle_steps VALUES (?,?,?,?,?,?,?,?,?,?,?)", steps)

    db.execute("INSERT INTO derive_runs VALUES (?,?,?,?,?,?)",
               (int(datetime.now().timestamp() * 1000), service_date, len(by_trip), n_pass, n_seg, None))
    db.commit()
    return {"service_date": service_date, "trips": len(by_trip), "passages": n_pass, "segments": n_seg}


SLOT_MS = 5 * 60 * 1000
WAIT_CAP_S = 3600       # a rider still waiting when the day's data ends counts as waiting this long
GAP_SETTLE_MS = 30 * 60 * 1000  # after a collector gap, trips already under way are missing their early stops
MIN_SLOT_SHARE = 0.5    # a slot needs at least half of its 5 minutes of service per stop to count
EDGE_STOPS = 2          # stops past a trip's first/last observed passage that inherit its delay


def mean_wait(times: list[int], a: int, b: int, cap_ms: int) -> float:
    """Mean wait (ms) until the next time in sorted `times` for someone arriving uniformly in [a, b)."""
    total, t, i = 0.0, a, bisect_right(times, a - 1)
    while t < b:
        nxt = times[i] if i < len(times) else None
        if nxt is None or nxt - t > cap_ms:  # no bus within the cap: wait capped
            end = b if nxt is None else min(b, nxt - cap_ms)
            total += (end - t) * cap_ms
            t = end
            continue
        end = min(b, nxt)
        total += (end - t) * (nxt - t + nxt - end) / 2
        t = end
        i += 1
    return total / (b - a)


def _wait_times(db: sqlite3.Connection, cfg: Config, since_ms: int):
    """Per (route, direction, stop): sorted scheduled and actual boarding times (epoch ms), and each service
    day's span of scheduled times there (riders are only counted while the stop has service).

    Actual times are every tracked passage, whichever trip or bus it was: a rider takes the next bus.
    Stops a tracked trip passed between two GPS-observed stops get the delay interpolated between them.
    A trip's last stop is left out (arriving at the terminal isn't a boarding chance; the next trip's
    first stop is)."""
    meta = {r["trip_id"]: (r["route_id"], r["direction_id"] or "")
            for r in db.execute("SELECT trip_id, route_id, direction_id FROM ref_trips ORDER BY day")}
    sched: dict[tuple, list] = defaultdict(list)
    actual: dict[tuple, list] = defaultdict(list)
    spans: dict[tuple, dict[int, list]] = defaultdict(dict)
    first_dep: dict[tuple, list] = defaultdict(list)  # (route, dir) -> [(first departure, tracked?)]
    observed = defaultdict(dict)
    for r in db.execute("SELECT service_date, trip_id, seq, actual_time, delay_s FROM stop_passages "
                        "WHERE service_date >= ?", (since_ms - 86_400_000,)):
        observed[(r["service_date"], r["trip_id"])][r["seq"]] = r["delay_s"]
    trips = defaultdict(list)
    for r in db.execute("SELECT trip_id, service_date, seq, stop_id, arrival_time, departure_time FROM stop_times "
                        "WHERE service_date >= ? ORDER BY trip_id, service_date, seq", (since_ms - 86_400_000,)):
        trips[(r["service_date"], r["trip_id"])].append(r)
    for (sd, trip_id), stops in trips.items():
        if trip_id not in meta or len(stops) < 2:
            continue
        route, d = meta[trip_id]
        base, _ = schedule_base(sd, route, cfg)
        obs = observed.get((sd, trip_id), {})
        seqs = sorted(obs)
        first = stops[0]
        first_dep[(route, d)].append((base + int(first["departure_time"] or first["arrival_time"]) * 1000, bool(obs)))
        for j, st in enumerate(stops[:-1]):
            sec = st["departure_time"] if j == 0 else st["arrival_time"]
            if sec is None:
                continue
            t_s = base + int(sec) * 1000
            key = (route, d, st["stop_id"])
            sched[key].append(t_s)
            span = spans[key].setdefault(sd, [t_s, t_s])
            span[0], span[1] = min(span[0], t_s), max(span[1], t_s)
            if st["seq"] in obs:
                delay = obs[st["seq"]]
            elif not seqs:
                continue  # no bus tracked on this trip
            else:
                k = bisect_right(seqs, st["seq"])
                if k in (0, len(seqs)):
                    # Before the first or after the last observed stop. Tracking often picks a trip up only
                    # after it leaves the terminal, so carry the nearest delay up to EDGE_STOPS stops out;
                    # farther than that, the bus may not have run that part, so leave it unknown.
                    near = seqs[0] if k == 0 else seqs[-1]
                    if abs(st["seq"] - near) > EDGE_STOPS:
                        continue
                    delay = obs[near]
                else:
                    lo, hi = seqs[k - 1], seqs[k]
                    delay = obs[lo] + (obs[hi] - obs[lo]) * (st["seq"] - lo) / (hi - lo)
            actual[key].append(int(t_s + delay * 1000))
    for v in list(sched.values()) + list(actual.values()):
        v.sort()
    return sched, actual, spans, first_dep


def rebuild_intervals(db: sqlite3.Connection, cfg: Config, since_ms: int) -> int:
    """5-minute route/direction slots (submitted proposal).

    The observation is excess wait: for a rider who shows up at a random moment in the slot, how much
    longer they wait for the next bus than the timetable promises, averaged over the route's stops. A bus
    running a full headway behind is then not "late" once the gaps between buses are back to normal, but
    bunching and missing buses are. Each trip's own schedule delay is kept alongside (delay_level)."""
    since_ms = since_ms // SLOT_MS * SLOT_MS
    sched, actual, spans, first_dep = _wait_times(db, cfg, since_ms)
    gaps = [(r[0] - SLOT_MS, r[1] + GAP_SETTLE_MS) for r in db.execute("SELECT start_ms, end_ms FROM gaps")]
    first_poll = db.execute("SELECT MIN(requested_at) FROM polls").fetchone()[0]
    if first_poll:  # collection start works like the end of a gap
        gaps.append((0, first_poll + GAP_SETTLE_MS))
    # A rider needs the next bus to be in the data: stop short of the newest poll (the next run fills it in).
    last_poll = db.execute("SELECT MAX(requested_at) FROM polls WHERE kind = 'tfr' AND error IS NULL").fetchone()[0]
    covered_until = (last_poll or 0) - GAP_SETTLE_MS
    cap = WAIT_CAP_S * 1000

    # Per route/direction/slot accumulators: [weighted sched, weighted actual, weight, n_stops, worst excess]
    acc: dict[tuple, list] = defaultdict(lambda: [0.0, 0.0, 0.0, 0, -math.inf])
    for key, s_times in sched.items():
        route, d, _ = key
        a_times = actual.get(key, [])
        for first, last in spans[key].values():
            # After the day's last tracked bus at this stop the wait isn't long, it's "no more service";
            # leave those riders out (a trip that never ran still shows in missed_trips).
            i = bisect_right(a_times, last + 3 * 3600 * 1000)
            if not i or a_times[i - 1] < first:
                continue  # no bus tracked here all day: unknown, not a long wait
            lo, hi = max(first, since_ms), min(last, covered_until, a_times[i - 1])
            slot = lo // SLOT_MS * SLOT_MS
            while slot < hi:
                a, b = max(slot, lo), min(slot + SLOT_MS, hi)
                if b > a and not any(g0 < b and a < g1 for g0, g1 in gaps):
                    ws, wa = mean_wait(s_times, a, b, cap), mean_wait(a_times, a, b, cap)
                    w = (b - a) / SLOT_MS
                    x = acc[(route, d, slot)]
                    x[0] += ws * w
                    x[1] += wa * w
                    x[2] += w
                    x[3] += 1
                    x[4] = max(x[4], (wa - ws) / 1000)
                slot += SLOT_MS

    missed: dict[tuple, int] = defaultdict(int)
    for (route, d), deps in first_dep.items():
        for t, tracked in deps:
            if not tracked and since_ms <= t <= covered_until:
                missed[(route, d, t // SLOT_MS * SLOT_MS)] += 1

    rows = db.execute(
        "SELECT p.route_id, COALESCE(p.direction_id,'') AS dir, p.actual_time, p.delay_s, s.gained_s "
        "FROM stop_passages p LEFT JOIN segments s ON s.service_date = p.service_date AND s.trip_id = p.trip_id "
        "AND s.seq = p.seq WHERE p.actual_time >= ?", (since_ms,)).fetchall()
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["route_id"], r["dir"], r["actual_time"] // SLOT_MS * SLOT_MS)].append((r["delay_s"], r["gained_s"]))

    db.execute("DELETE FROM intervals_5min WHERE slot_start >= ?", (since_ms,))
    out = []
    for key in sorted(set(groups) | set(acc)):
        route, d, start = key
        vals = groups.get(key, [])
        delays = sorted(v[0] for v in vals)
        gained = [v[1] for v in vals if v[1] is not None]
        if delays:
            mean = statistics.fmean(delays)
            dstats = (len(delays), mean, statistics.median(delays), delays[min(len(delays) - 1, int(0.9 * len(delays)))],
                      delays[-1], statistics.fmean(gained) if gained else None, level_of(mean, cfg.deviation_levels))
        else:
            dstats = (0, None, None, None, None, None, None)
        x = acc.get(key)
        if x and x[2] >= MIN_SLOT_SHARE * x[3]:  # skip slivers at the very start/end of service
            ws, wa = x[0] / x[2] / 1000, x[1] / x[2] / 1000
            wstats = (x[3], ws, wa, wa - ws, x[4], missed.get(key, 0), level_of(wa - ws, cfg.wait_levels))
        else:
            wstats = (0, None, None, None, None, missed.get(key, 0), None)
        out.append((route, d, start, datetime.fromtimestamp(start / 1000, cfg.timezone).strftime("%Y-%m-%d %H:%M"),
                    *dstats, *wstats))
    db.executemany(f"INSERT OR REPLACE INTO intervals_5min VALUES ({','.join('?' * 18)})", out)
    db.commit()
    return len(out)


def rebuild_gaps(db: sqlite3.Connection, cfg: Config, since_ms: int) -> int:
    """Stretches with no successful trips-for-route poll for longer than expected.

    Overnight the collector polls every KF_IDLE_POLL_INTERVAL_S, so a gap is any wait between
    successful polls longer than that (plus a minute of slack), not a missing minute."""
    limit = (max(cfg.poll_interval_s, cfg.idle_poll_interval_s) + 60) * 1000
    # Poll timestamps directly (the cycle counter restarts with the collector, so it can't be grouped on).
    times = [r[0] for r in db.execute(
        "SELECT requested_at FROM polls WHERE kind = 'tfr' AND error IS NULL "
        "AND http_status BETWEEN 200 AND 299 AND requested_at >= ? ORDER BY requested_at", (since_ms,))]
    now = int(datetime.now().timestamp() * 1000)
    if times:
        times.append(now)  # a gap still going on now
    db.execute("DELETE FROM gaps WHERE start_ms >= ?", (since_ms,))
    gaps = [(a, b, round((b - a) / 60000)) for a, b in zip(times, times[1:]) if b - a > limit]
    db.executemany("INSERT OR REPLACE INTO gaps VALUES (?,?,?)", gaps)
    db.commit()
    return len(gaps)


def run(db: sqlite3.Connection, cfg: Config, days: int = 2, full: bool = False) -> list[dict]:
    cols = {r[1] for r in db.execute("PRAGMA table_info(intervals_5min)")}
    old_layout = bool(cols) and "excess_wait_s" not in cols
    if old_layout:  # pre-1.3.0 intervals: derived only, so drop and rebuild them for every day
        db.execute("DROP TABLE intervals_5min")
    db.executescript(DERIVED_SCHEMA)
    if full:
        dates = [r[0] for r in db.execute("SELECT DISTINCT service_date FROM trip_status WHERE service_date IS NOT NULL")]
        since = 0
    else:
        cutoff = int((datetime.now(cfg.timezone) - timedelta(days=days)).timestamp() * 1000)
        dates = [r[0] for r in db.execute(
            "SELECT DISTINCT service_date FROM trip_status WHERE service_date >= ?", (cutoff - 86_400_000,))]
        since = cutoff - 86_400_000
    results = [derive_day(db, cfg, d) for d in sorted(d for d in dates if d)]
    rebuild_gaps(db, cfg, since)
    rebuild_intervals(db, cfg, 0 if old_layout else since)
    return results
