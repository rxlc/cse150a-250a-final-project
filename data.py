"""Build HMM observation sequences per route per service day using stop_passages.csv and stop_times.csv

For every stop, each gap between two consecutive buses is compared with the scheduled gap:
    r = actual gap / scheduled gap
and seperated into bins:
    0 short (bunched)   r < 0.5
    1 regular           0.5 <= r < 1.25
    2 long              1.25 <= r < 2
    3 very long         r >= 

Outputs, in ./output:
    <route>.npz            one integer array per service day, keyed "YYYY-MM-DD"

All times are epoch milliseconds (integers) until they are printed
"""

from pathlib import Path

from collections import defaultdict

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
ROUTES = {"UCSD_1040": "inside_loop", "UCSD_1030": "outside_loop", "UCSD_1050": "mesa_loop"}
STEP_MS = 5 * 60 * 1000
TZ = "America/Los_Angeles"

EXCLUDE_DAYS = {"2026-11-01", "2026-11-11"} # daylight savings & Veterans day

def load(name):
    """Read data/<name>.csv with ids read as text so they keep leading zeros"""
    ids = {c: str for c in ["trip_id", "stop_id", "route_id", "direction_id"]}
    return pd.read_csv(DATA_DIR / f"{name}.csv", dtype=ids)

def local_day(ms):
    """Epoch ms -> local date 'YYYY-MM-DD'"""
    t = pd.to_datetime(ms, unit="ms", utc=True)
    return t.dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")

def last_stops(stop_times):
    """Returns table with highest stop number (last stop) per trip_id and service_date"""
    return (stop_times.groupby(["trip_id", "service_date"], as_index=False)["seq"]
        .max().rename(columns={"seq": "last_seq"}))

def clean(table):
    """Takes in scheduled or actual tables. Drop rows without a time, label the day, and drop excluded days"""
    table = table.dropna(subset=["time"]).copy()

    if table.empty:
        return pd.DataFrame(columns=["day", "stop_id", "time"])
    
    table["time"] = table["time"].astype("int64")
    table["day"] = local_day(table["time"])

    weekday = pd.to_datetime(table["day"]).dt.dayofweek < 5
    excluded = table["day"].isin(EXCLUDE_DAYS)
    table = table[weekday & ~excluded]

    return table[["day", "stop_id", "time"]]

def scheduled_times(route_id, stop_times, ref_trips, last):
    """Returns table when the route's buses are scheduled at each stop (leaving out last stop)"""
    trips = set(ref_trips.loc[ref_trips["route_id"] == route_id, "trip_id"])

    scheduled_table = stop_times[stop_times["trip_id"].isin(trips)].copy()
    seconds = scheduled_table["arrival_time"].where(scheduled_table["seq"] != 0, scheduled_table["departure_time"]) # use departure_time at first stop instead of arrival_time
    scheduled_table["time"] = scheduled_table["service_date"] + seconds * 1000

    scheduled_table = scheduled_table.merge(last, on=["trip_id", "service_date"])
    scheduled_table = scheduled_table[scheduled_table["seq"] < scheduled_table["last_seq"]]
    return clean(scheduled_table)

def actual_times(route_id, stop_passages, last):
    """Returns table when the route's bus actually reached each stop (leaving out last stop)"""
    actual_table = stop_passages[stop_passages["route_id"] == route_id].merge(last, on=["trip_id", "service_date"])

    actual_table = actual_table[actual_table["seq"] < actual_table["last_seq"]].copy()
    actual_table["time"] = actual_table["actual_time"]
    return clean(actual_table)

def times_by_stop(table, unique=False):
    """Returns a dict {(day, stop_id): sorted list of times}
    Ex: (2026-10-28 UCSD_11902): ['07:21', '07:40', '07:54', ...]"""
    stops_dict = {}

    for (day, stop), group in table.groupby(["day", "stop_id"]):
        times = group["time"].tolist()
        if unique:
            times = set(times)
        stops_dict[(day, stop)] = sorted(times)
    return stops_dict

def scheduled_gap(scheduled_times, t):
    """The scheduled gap that time t falls in. A bus arriving after the last scheduled time
    is compared with the day's last scheduled gap. None if there is no gap to compare with."""
    if len(scheduled_times) < 2 or t <= scheduled_times[0]:
        return None
    
    for before, after in zip(scheduled_times[:-1], scheduled_times[1:]):
        if before < t <= after:
            return after - before

    return scheduled_times[-1] - scheduled_times[-2]

def bin_ratio(r):
    """Actual gap / scheduled gap -> 0 short, 1 regular, 2 long, 3 very long."""
    if r < 0.5:
        return 0
    elif r < 1.25:
        return 1
    elif r < 2.0:
        return 2
    else:
        return 3

def headways(actual_by_stop, sched_by_stop):
    """Returns dict {day: list of (time the gap ended, bin)}, from every stop of the route."""
    gaps_by_day = defaultdict(list)
    skipped = 0

    for (day, stop), arrivals in actual_by_stop.items():
        sched = sched_by_stop.get((day, stop), [])

        for prev, cur in zip(arrivals[:-1], arrivals[1:]):
            H = scheduled_gap(sched, cur)
            if H is None:
                skipped += 1
                continue
            gaps_by_day[day].append((cur, bin_ratio((cur - prev) / H)))

    return gaps_by_day, skipped

def day_sequence(day_gaps, first_time, last_time):
    """The day's observation array, and the time its step 0 starts.
    Step t covers [start + t * 5 min, start + (t + 1) * 5 min)."""
    start = first_time // STEP_MS * STEP_MS # round down to a 5-minute mark
    n_steps = (last_time - start) // STEP_MS + 1

    obs = np.full(n_steps, -1)
    for cur, b in day_gaps:
        t = (cur - start) // STEP_MS
        obs[t] = max(obs[t], b)
    
    return obs, start

def build_route(route_id, stop_passages, stop_times, ref_trips, last):
    """Returns {day: observation array} and {day: start time} for one route."""
    scheduled_table = scheduled_times(route_id, stop_times, ref_trips, last)
    actual_table = actual_times(route_id, stop_passages, last)

    actual_by_stop = times_by_stop(actual_table)
    scheduled_by_stop = times_by_stop(scheduled_table, unique=True)
    gaps_by_day, skipped = headways(actual_by_stop, scheduled_by_stop)

    seqs, starts = {}, {}
    for day in sorted(scheduled_table["day"].unique()):
        day_gaps = gaps_by_day[day]
        day_schedule = scheduled_table.loc[scheduled_table["day"] == day, "time"]

        first = day_schedule.min()
        last_time = max([day_schedule.max()] + [cur for cur, _ in day_gaps])

        seqs[day], starts[day] = day_sequence(day_gaps, first, last_time)
    return seqs, starts, skipped

def show(obs, start, from_clock="7:00", n=12):
    """Print n steps of a sequence with their clock times above them."""
    times = [pd.Timestamp(start + t * STEP_MS, unit="ms", tz="UTC").tz_convert(TZ).strftime("%H:%M")
        for t in range(len(obs))]
    first = next((t for t, c in enumerate(times) if c >= from_clock), 0)

    print("  " + "  ".join(times[first:first + n]))
    print("  " + "  ".join(f"{v:>5}" for v in obs[first:first + n]))
    
    
def main():
    stop_passages = load("stop_passages")
    stop_times = load("stop_times")
    ref_trips = load("ref_trips")
    assert len(load("gaps")) == 0

    OUT_DIR.mkdir(exist_ok=True)

    last = last_stops(stop_times)
    for route_id, name in ROUTES.items():
        seqs, starts, skipped = build_route(route_id, stop_passages, stop_times, ref_trips, last)
        
        np.savez(OUT_DIR / f"{name}.npz", **seqs)
        print(f"{name}: {len(seqs)} days saved, {skipped} gaps skipped")

        # if name == "outside_loop":
        #     for day, clock_from in [("2026-11-02", "7:15")]:
        #         if day in seqs:
        #             print(f"  {day} from {clock_from}:")
        #             show(seqs[day], starts[day], clock_from)

if __name__ == "__main__":
    main()