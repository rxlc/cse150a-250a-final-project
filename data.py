from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
TZ = "America/Los_Angeles"

def load(name):
    ids = {c: str for c in ["trip_id", "stop_id", "route_id", "direction_id"]}
    return pd.read_csv(DATA_DIR / f"{name}.csv", dtype=ids)

def to_local(ms):
    return pd.to_datetime(ms, unit="ms", utc=True).dt.tz_convert(TZ)


if __name__ == "__main__":
    passages = load("stop_passages")
    stop_times = load("stop_times")
    routes = load("ref_routes").drop_duplicates("route_id")

    print("stop_passages:", passages.shape)
    print("stop_times:   ", stop_times.shape)

    print("\nRoutes:")
    print(routes[["route_id", "short_name", "long_name"]].to_string(index=False))

    print("\nStop passages per route:")
    print(passages["route_id"].value_counts())

    times = to_local(passages["actual_time"])
    print("\nFirst passage:", times.min())
    print("Last passage: ", times.max())

    print("\nFirst rows of stop_passages:")
    print(passages.head())