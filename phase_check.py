""" Prints, for each route, the share of each level by the interval's start minute mod 15
(rows: 0 = :00/:15/:30/:45, 5 = :05/:20/..., 10 = :10/:25/...), from the saved 4-level
sequences (before any merge), and the number of intervals in each row.

    python phase_check.py --obs ratio
    python phase_check.py --obs delay
"""

import argparse

import numpy as np
import pandas as pd

from data import OUT_DIR, ROUTES, STEP_MS, TZ, load_sequences

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", choices=["level", "ratio", "delay"], default="level")
    args = ap.parse_args()

    for name in ROUTES.values():
        seqs = load_sequences(name, args.obs)
        starts = pd.read_csv(OUT_DIR / f"{name}_{args.obs}_days.csv").set_index("day")["start_ms"]
        rows = []
        for day, o in seqs.items():
            t = pd.to_datetime(starts[day] + np.arange(len(o)) * STEP_MS, unit="ms", utc=True).tz_convert(TZ)
            rows.append(pd.DataFrame({"phase": t.minute % 15, "level": o}))
        df = pd.concat(rows, ignore_index=True)
        df = df[df["level"] >= 0]

        print(f"=== {name} ({args.obs}): share of each level by minute mod 15")
        table = pd.crosstab(df["phase"], df["level"], normalize="index").round(3)
        table["intervals"] = df["phase"].value_counts().sort_index()
        print(table.to_string())
        print()

if __name__ == "__main__":
    main()
