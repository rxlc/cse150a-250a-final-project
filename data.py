"""
Build HMM observation sequences per route per weekday, for three observations:

  level  (core)       excess rider wait from the collector's intervals_5min.csv, binned
                      0 shorter, 1 normal, 2 longer, 3 much longer
  ratio  (ablation)   gap between consecutive buses at each stop / scheduled gap, binned
                      0 short (r < 0.5), 1 regular (< 1.25), 2 long (< 2), 3 very long;
                      each 5-minute step takes the worst gap that ended in it
  delay  (ablation)   mean trip delay from intervals_5min.csv (delay_level), binned
                      0 early, 1 on time, 2 late, 3 very late

-1 marks a missing step in every observation.

Outputs, in ./output, for each route and observation:
    <route>_<obs>.npz        one integer array per service day, keyed "YYYY-MM-DD"
    <route>_<obs>_days.csv   one row per day: start time, steps, missing share, share of each bin

All times are epoch milliseconds (integers) until they are printed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import OBSERVATIONS, TEST_DAYS

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
ROUTES = {"UCSD_1040": "inside_loop", "UCSD_1030": "outside_loop", "UCSD_1050": "mesa_loop"}
STEP_MS = 5 * 60 * 1000
TZ = "America/Los_Angeles"

EXCLUDE_DAYS = {"2026-11-01", "2026-11-11", "2026-11-18"} # daylight savings & Veterans day & Nov 18
# Nov 18: the OneBusAway feed reported 0 active buses on every route (UCSD and MTS) from about
# 13:20 to 13:45, so that afternoon's long waits are missing data, not real delays

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

def keep_days(table):
    """Weekdays only, minus the excluded days. Used by every observation."""
    weekday = pd.to_datetime(table["day"]).dt.dayofweek < 5
    excluded = table["day"].isin(EXCLUDE_DAYS)
    return table[weekday & ~excluded]

def clean(table):
    """Takes in scheduled or actual tables. Drop rows without a time, label the day, and drop excluded days"""
    table = table.dropna(subset=["time"]).copy()

    if table.empty:
        return pd.DataFrame(columns=["day", "stop_id", "time"])
    
    table["time"] = table["time"].astype("int64")
    table["day"] = local_day(table["time"])
    table = keep_days(table)

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

GAP_COLUMNS = ["day", "stop_id", "prev", "cur", "h_min", "H_min", "r", "bin"]

def headways(actual_by_stop, sched_by_stop):
    """Returns a table with one row per gap between consecutive buses at a stop, from every
    stop of the route, and the number of gaps skipped (no scheduled gap to compare with)."""
    rows = []
    skipped = 0

    for (day, stop), arrivals in actual_by_stop.items():
        sched = sched_by_stop.get((day, stop), [])

        for prev, cur in zip(arrivals[:-1], arrivals[1:]):
            H = scheduled_gap(sched, cur)
            if H is None:
                skipped += 1
                continue
            r = (cur - prev) / H
            rows.append({"day": day, "stop_id": stop, "prev": prev, "cur": cur,
                         "h_min": round((cur - prev) / 60000, 2), "H_min": round(H / 60000, 2),
                         "r": round(r, 3), "bin": bin_ratio(r)})

    return pd.DataFrame(rows, columns=GAP_COLUMNS), skipped

def day_sequence(day_gaps, first_time, last_time):
    """The day's observation array, and the time its step 0 starts.
    Step t covers [start + t * 5 min, start + (t + 1) * 5 min)."""
    start = first_time // STEP_MS * STEP_MS # round down to a 5-minute mark
    n_steps = (last_time - start) // STEP_MS + 1

    obs = np.full(n_steps, -1)
    for cur, b in zip(day_gaps["cur"], day_gaps["bin"]):
        t = (cur - start) // STEP_MS
        obs[t] = max(obs[t], b)

    return obs, start

def clock(ms):
    """Epoch ms -> 'HH:MM' local time, for reading."""
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert(TZ).strftime("%H:%M")

def day_summary(day, obs, start, n_headways=None):
    """One row describing a day: when it starts and ends, how many steps, and the share of
    steps in each bin. n_headways (ratio only) is the number of gaps behind the day."""
    row = {"day": day, "start": clock(start), "start_ms": start,
           "end": clock(start + (len(obs) - 1) * STEP_MS), "steps": len(obs),
           "missing": round(float(np.mean(obs == -1)), 3)}
    if n_headways is not None:
        row["headways"] = n_headways
    for b in range(4):
        row[f"bin{b}"] = round(float(np.mean(obs == b)), 3)
    return row

def build_route(route_id, stop_passages, stop_times, ref_trips, last):
    """Returns {day: observation array}, the table of gaps, a one-row-per-day summary,
    and the number of skipped gaps, for one route."""
    scheduled_table = scheduled_times(route_id, stop_times, ref_trips, last)
    actual_table = actual_times(route_id, stop_passages, last)

    actual_by_stop = times_by_stop(actual_table)
    scheduled_by_stop = times_by_stop(scheduled_table, unique=True)
    gaps, skipped = headways(actual_by_stop, scheduled_by_stop)

    seqs, summary = {}, []
    for day in sorted(scheduled_table["day"].unique()):
        day_gaps = gaps[gaps["day"] == day]
        day_schedule = scheduled_table.loc[scheduled_table["day"] == day, "time"]

        first = day_schedule.min()
        last_time = max([day_schedule.max()] + day_gaps["cur"].tolist())

        seqs[day], start = day_sequence(day_gaps, first, last_time)
        summary.append(day_summary(day, seqs[day], start, len(day_gaps)))

    return seqs, gaps, pd.DataFrame(summary), skipped

def level_sequences(route_id, column="level"):
    """Sequences from the collector's intervals_5min.csv: {day: array}, {day: start time}.
    column="level" is excess wait (core); column="delay_level" is trip delay (ablation).
    Each day runs from its first to its last slot with a value. A slot with an empty value,
    or with no row at all, is -1, so the grid keeps its true 5-minute spacing."""
    iv = load("intervals_5min")
    iv = iv[iv["route_id"] == route_id].dropna(subset=[column]).copy()
    iv["day"] = local_day(iv["slot_start"])
    iv = keep_days(iv)

    seqs, starts = {}, {}
    for day, g in iv.groupby("day"):
        start = g["slot_start"].min()
        obs = np.full((g["slot_start"].max() - start) // STEP_MS + 1, -1)
        obs[((g["slot_start"] - start) // STEP_MS).to_numpy()] = g[column].astype(int).to_numpy()
        seqs[day], starts[day] = obs, start
    return seqs, starts

def save(name, obs_name, seqs, summary):
    """Write <route>_<obs>.npz and <route>_<obs>_days.csv."""
    np.savez(OUT_DIR / f"{name}_{obs_name}.npz", **seqs)
    summary.to_csv(OUT_DIR / f"{name}_{obs_name}_days.csv", index=False)

def load_sequences(name, obs_name="level"):
    """For the HMM and baselines: {day: array} for one route and observation
    ('level', 'ratio' or 'delay')."""
    return dict(np.load(OUT_DIR / f"{name}_{obs_name}.npz"))

def print_sequence(name, day, from_clock="00:00", n=12, obs_name="level"):
    """Print n steps of a saved sequence, with clock times above the values.
    from_clock must be zero-padded, e.g. "07:15"."""
    seqs = np.load(OUT_DIR / f"{name}_{obs_name}.npz")
    days = pd.read_csv(OUT_DIR / f"{name}_{obs_name}_days.csv").set_index("day")
    if day not in seqs:
        print(f"{name} {obs_name} {day}: not in the data")
        return

    obs, start = seqs[day], int(days.loc[day, "start_ms"])
    times = [clock(start + t * STEP_MS) for t in range(len(obs))]
    first = next((t for t, c in enumerate(times) if c >= from_clock), None)
    if first is None:
        print(f"  (no steps at or after {from_clock})")
        return
    print("  " + "  ".join(times[first:first + n]))
    print("  " + "  ".join(f"{v:>5}" for v in obs[first:first + n]))

def merge_0_1_levels(o):
    return np.where(o < 0, -1, np.maximum(o, 1) - 1)

def load_route(name, obs_name="level"):
    """For the HMM and baselines: {day: levels, -1 missing} and {day: start of step 0 in epoch ms}
    for one observation. Levels 0 and 1 are merged if config.OBSERVATIONS says so (for "level":
    0 normal or shorter, 1 longer, 2 much longer)."""
    _, merge = OBSERVATIONS[obs_name]
    seqs = load_sequences(name, obs_name)
    if merge:
        seqs = {day: merge_0_1_levels(o) for day, o in seqs.items()}
    days = pd.read_csv(OUT_DIR / f"{name}_{obs_name}_days.csv")
    starts = dict(zip(days["day"], days["start_ms"].astype("int64")))
    return seqs, starts

def split_days(days):
    """(train days, test days): test on config.TEST_DAYS, train on the rest. The same for every model."""
    days = sorted(days)
    assert all(d in days for d in TEST_DAYS), "a test day is missing from the data"
    return [d for d in days if d not in TEST_DAYS], list(TEST_DAYS)

def main():
    stop_passages = load("stop_passages")
    stop_times = load("stop_times")
    ref_trips = load("ref_trips")

    OUT_DIR.mkdir(exist_ok=True)

    assert len(load("gaps")) == 0

    last = last_stops(stop_times)

    for route_id, name in ROUTES.items():
        built = {}

        # core and second ablation
        for obs_name, column in [("level", "level"), ("delay", "delay_level")]:
            seqs, starts = level_sequences(route_id, column)
            summary = pd.DataFrame([day_summary(d, seqs[d], starts[d]) for d in sorted(seqs)])
            built[obs_name] = (seqs, summary)

        # first ablation: headway ratio
        seqs, gaps, summary, skipped = build_route(route_id, stop_passages, stop_times, ref_trips, last)
        built["ratio"] = (seqs, summary)

        if not any(seqs for seqs, _ in built.values()):
            print(f"{name}: no data for {route_id} (check the id in ref_routes.csv)")
            continue

        print(f"\n===== {name} =====")
        for obs_name, (seqs, summary) in built.items():
            if not seqs:
                print(f"{obs_name}: no days")
                continue
            save(name, obs_name, seqs, summary)
            print(f"\n{obs_name}: {len(seqs)} days")
            print(summary.drop(columns="start_ms").to_string(index=False))
        if len(gaps):
            print(f"ratio: {len(gaps):,} gaps ({skipped} skipped), {(gaps['bin'] == 1).mean():.1%} regular")

if __name__ == "__main__":
    main()
