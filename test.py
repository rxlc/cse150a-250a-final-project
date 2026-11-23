import numpy as np, pandas as pd
from data import load_sequences, OUT_DIR, STEP_MS, TZ

seqs = load_sequences("outside_loop")
starts = pd.read_csv(OUT_DIR / "outside_loop_level_days.csv").set_index("day")["start_ms"]
rows = []
for day, o in seqs.items():
    t = pd.to_datetime(starts[day] + np.arange(len(o)) * STEP_MS, unit="ms", utc=True).tz_convert(TZ)
    rows.append(pd.DataFrame({"phase": t.minute % 15, "level": o}))

df = pd.concat(rows, ignore_index=True)
print(pd.crosstab(df["phase"], df["level"], normalize="index").round(3))
print(df["phase"].value_counts().sort_index())    # how many intervals in each row