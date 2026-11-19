"""Build HMM observation sequences per route per service day

For every stop, each gap between two consecutive buses is compared with the scheduled gap:
    r = actual gap / scheduled gap
and seperated into bins:
    0 short (bunched)   r < 0.5
    1 regular           0.5 <= r < 1.25
    2 long              1.25 <= r < 2
    3 very long         r >= 

Outputs, in ./output:
    <route>.npz            one integer array per service day, keyed "YYYY-MM-DD"
    <route>_headways.csv   one row per gap, for checking and debugging
    <route>_days.csv       one row per day: start time, steps, missing share, share of each bin

All times are epoch milliseconds (integers) until they are printed
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
ROUTES = {"UCSD_1040": "inside_loop", "UCSD_1030": "outside_loop", "UCSD_1050": "mesa_loop"}
EXCLUDE_DAYS = {"2026-11-01", "2026-11-11"} # daylight savings & Veterans day
STEP_MS = 5 * 60 * 1000 
TZ = "America/Los_Angeles"
OUT_DIR = Path(__file__).parent / "output"

def load(name):
    ids = {c: str for c in ["trip_id", "stop_id", "route_id", "direction_id"]}
    return pd.read_csv(DATA_DIR / f"{name}.csv", dtype=ids)

def service_day(service_date_ms):
    """Turn service_date column (epoch ms) into 'YYYY-MM-DD' local dates."""
    t = pd.to_datetime(service_date_ms + 12 * 3600 * 1000, unit="ms", utc=True)
    return t.dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")

def clock(ms):
    """Epoch ms -> HH:MM for printing"""
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert(TZ).strftime("%H:%M")

def keep_day(day):
    if day in EXCLUDE_DAYS:
        return False
    if pd.Timestamp(day).dayofweek >= 5:
        return False
    return True

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


def scheduled_gap(sched_times, t):
    """The scheduled gap that time t falls in: walk through consecutive scheduled times
    and return the length of the pair that t falls between.
    A bus arriving after the last scheduled time (the day's last bus, running late) is
    compared with the day's last scheduled gap. None if there is no gap to compare with."""
    if len(sched_times) < 2 or t <= sched_times[0]:
        return None
    for before, after in zip(sched_times[:-1], sched_times[1:]):
        if before < t <= after:
            return after - before
    return sched_times[-1] - sched_times[-2]


def overlaps_gap(start, end, gaps):
    """True if the interval [start, end] touches a collector outage (padded by one step)."""
    for gap_start, gap_end in gaps:
        if start < gap_end + STEP_MS and end > gap_start - STEP_MS:
            return True
    return False

def route_tables(route_id, passages, stop_times, ref_trips):
    """Two tables with columns day, stop_id, time (epoch ms): when buses were scheduled at
    each stop, and when they actually came. Each trip's last stop is left out of both."""

    trips = set(ref_trips.loc[ref_trips["route_id"] == route_id, "trip_id"])

    sched = stop_times[stop_times["trip_id"].isin(trips)].copy()
    seconds = sched["arrival_time"].where(sched["seq"] != 0, sched["departure_time"])
    sched["time"] = sched["service_date"] + seconds * 1000

    last = (sched.groupby(["trip_id", "service_date"], as_index=False)["seq"].max().rename(columns={"seq": "last_seq"}))
    sched = sched.merge(last, on=["trip_id", "service_date"])
    sched = sched[sched["seq"] < sched["last_seq"]]

    actual = passages[passages["route_id"] == route_id].merge(last, on=["trip_id", "service_date"])
    actual = actual[actual["seq"] < actual["last_seq"]].copy()
    actual["time"] = actual["actual_time"]

    tables = []
    for table in (sched, actual):
        table = table.dropna(subset=["time"]).copy()
        if table.empty:
            tables.append(pd.DataFrame(columns=["day", "stop_id", "trip_id", "time"]))
            continue
        table["time"] = table["time"].astype("int64")
        table["day"] = service_day(table["service_date"])
        table = table[table["day"].map(keep_day)]
        tables.append(table[["day", "stop_id", "trip_id", "time"]])
    return tables[0], tables[1]


def times_by_stop(table, unique=False):
    out = {}
    for (day, stop), group in table.groupby(["day", "stop_id"]):
        times = group["time"].tolist()
        if unique:
            times = set(times)
        out[(day, stop)] = sorted(times)
    return out

def headways(actual_by_stop, sched_by_stop, gaps):
    rows = []
    skipped = 0
    for (day, stop), arrivals in actual_by_stop.items():
        sched = sched_by_stop.get((day, stop), [])
        for prev, cur in zip(arrivals[:-1], arrivals[1:]):
            H = scheduled_gap(sched, cur)
            if H is None or overlaps_gap(prev, cur, gaps):
                skipped += 1
                continue
            r = (cur - prev) / H
            rows.append({"day": day, "stop_id": stop, "prev": prev, "cur": cur,
                         "h_min": round((cur - prev) / 60000, 2), "H_min": round(H / 60000, 2),
                         "r": round(r, 3), "bin": bin_ratio(r)})
    columns = ["day", "stop_id", "prev", "cur", "h_min", "H_min", "r", "bin"]
    return pd.DataFrame(rows, columns=columns), skipped


def day_sequence(day_headways, first_time, last_time, gaps):
    start = first_time // STEP_MS * STEP_MS
    n_steps = (last_time - start) // STEP_MS + 1
    obs = np.full(n_steps, -1)         

    for cur, b in zip(day_headways["cur"], day_headways["bin"]):
        t = (cur - start) // STEP_MS
        obs[t] = max(obs[t], b)

    for t in range(n_steps):
        step_start = start + t * STEP_MS
        if overlaps_gap(step_start, step_start + STEP_MS, gaps):
            obs[t] = -1
    return obs, start


def build_route(route_id, passages, stop_times, ref_trips, gaps):
    sched, actual = route_tables(route_id, passages, stop_times, ref_trips)
    hw, skipped = headways(times_by_stop(actual), times_by_stop(sched, unique=True), gaps)

    seqs, summary = {}, []
    hw["step"] = -1
    for day in sorted(sched["day"].unique()):
        day_sched = sched[sched["day"] == day]
        day_actual = actual[actual["day"] == day]
        day_hw = hw[hw["day"] == day]
        first = day_sched["time"].min()
        last = max(day_sched["time"].max(), day_actual["time"].max() if len(day_actual) else 0)

        obs, start = day_sequence(day_hw, first, last, gaps)
        seqs[day] = obs
        hw.loc[hw["day"] == day, "step"] = (day_hw["cur"] - start) // STEP_MS

        row = {"day": day, "start": clock(start), "start_ms": start, "steps": len(obs),
               "missing": round(float(np.mean(obs == -1)), 3), "headways": len(day_hw)}
        for b, name in enumerate(["short", "regular", "long", "very_long"]):
            row[name] = round(float(np.mean(obs == b)), 3)
        summary.append(row)
    return seqs, hw, pd.DataFrame(summary), skipped


def check_schedule_formula(passages, stop_times):
    st = stop_times.copy()
    seconds = st["arrival_time"].where(st["seq"] != 0, st["departure_time"])
    st["time"] = st["service_date"] + seconds * 1000
    m = st.merge(passages, on=["trip_id", "service_date", "seq"])
    m = m[~service_day(m["service_date"]).isin(EXCLUDE_DAYS)]
    print(f"Schedule formula check: {(m['time'] == m['scheduled_time']).mean():.2%} of "
          f"{len(m):,} rows match")


def print_sequence(name, day, from_clock="00:00", n=12):
    seqs = np.load(OUT_DIR / f"{name}.npz")
    days = pd.read_csv(OUT_DIR / f"{name}_days.csv").set_index("day")
    if day not in seqs:
        print(f"{name} {day}: not in the data")
        return
    obs, start = seqs[day], int(days.loc[day, "start_ms"])
    times = [clock(start + t * STEP_MS) for t in range(len(obs))]
    first = next((t for t, c in enumerate(times) if c >= from_clock), 0)
    print(f"{name} {day}:")
    print("  " + "  ".join(times[first:first + n]))
    print("  " + "  ".join(f"{v:>5}" for v in obs[first:first + n]))


def main():
    passages = load("stop_passages")
    stop_times = load("stop_times")
    ref_trips = load("ref_trips").drop_duplicates("trip_id")

    gaps = []
    if (DATA_DIR / "gaps.csv").exists():
        g = load("gaps")
        gaps = list(zip(g["start_ms"], g["end_ms"]))

    check_schedule_formula(passages, stop_times)
    OUT_DIR.mkdir(exist_ok=True)

    for route_id, name in ROUTES.items():
        seqs, hw, summary, skipped = build_route(route_id, passages, stop_times, ref_trips, gaps)

        np.savez(OUT_DIR / f"{name}.npz", **seqs)
        hw.to_csv(OUT_DIR / f"{name}_headways.csv", index=False)
        summary.to_csv(OUT_DIR / f"{name}_days.csv", index=False)
        print(f"\n{name}: {len(seqs)} days, {len(hw):,} headways ({skipped:,} skipped), "
              f"{(hw['bin'] == 1).mean():.1%} regular")
        print(summary.drop(columns="start_ms").to_string(index=False))

    print()
    print_sequence("outside_loop", "2026-11-02", "12:05")
    print_sequence("outside_loop", "2026-10-28", "16:00")


if __name__ == "__main__":
    main()