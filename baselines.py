"""Three baseline models for the bus wait HMM.

Each baseline is a simpler model that the HMM has to beat. All three are learned by
maximum likelihood, i.e. by counting and normalising, like the CPTs in class.

    independent   P(o_t)                 ignores time completely (does the past matter at all?)
    markov        P(o_t | o_{t-1})       uses the last level, no hidden states (do hidden states help?)
    time_of_day   P(o_t | hour of day)   uses only the clock (is it just the time of day?)

Every predict_* function returns one predicted distribution over the 4 levels for each
observed interval of a day. evaluate.py scores those predictions.

Data: the sequences that data.py writes to output/ (run `python data.py` first), one
integer per 5-minute interval, -1 = missing, for one of three observations:
    level (core): excess wait, 0 shorter, 1 normal, 2 longer, 3 much longer
    ratio, delay (ablations), see data.py
The split into training and test days is done here (split_days), because it is part of
the evaluation: every model, including the HMM, must learn from the training days only.

Run this file on its own to print the learned tables:
    python baselines.py
    python baselines.py --route mesa_loop --obs ratio
"""

import argparse

import numpy as np
import pandas as pd

from data import OUT_DIR, STEP_MS, TZ, load_sequences

ROUTES = ["inside_loop", "outside_loop", "mesa_loop"]
OBSERVATIONS = ["level", "ratio", "delay"]
LEVELS = ["level_0", "level_1", "level_2", "level_3"]  # names depend on the observation
K = len(LEVELS)
RES_DIR = OUT_DIR.parent / "results"


# ---------------------------------------------------------------- data (read data.py's output)

def load_route(name, obs="level"):
    """{day: array of levels, -1 = missing} and {day: start time of step 0 in ms},
    from output/<route>_<obs>.npz and output/<route>_<obs>_days.csv written by data.py."""
    seqs = load_sequences(name, obs)
    days = pd.read_csv(OUT_DIR / f"{name}_{obs}_days.csv")
    starts = dict(zip(days["day"], days["start_ms"].astype("int64")))
    return seqs, starts


def split_days(seqs, n_test):
    """Train on the earlier days and test on the last n_test days, so every model is
    tested on days it has never seen."""
    days = sorted(seqs)
    if n_test >= len(days):
        raise ValueError(f"only {len(days)} days, cannot hold out {n_test}")
    return days[:-n_test], days[-n_test:]


def hour_of_each_interval(day, n, starts):
    """Local hour of day (0-23) of each of the n intervals of a day."""
    t = pd.to_datetime(starts[day] + np.arange(n) * STEP_MS, unit="ms", utc=True)
    return t.tz_convert(TZ).hour.to_numpy()


# ---------------------------------------------------------------- 1. independent: P(o_t)

def fit_independent(train, seqs):
    """Count how often each level appears, add 1 to every count (so no level gets
    probability 0), and divide by the total."""
    counts = np.ones(K)
    for day in train:
        o = seqs[day]
        counts += np.bincount(o[o >= 0], minlength=K)
    return counts / counts.sum()


def predict_independent(p, levels):
    """The same distribution for every observed interval."""
    return np.tile(p, ((levels >= 0).sum(), 1))


# ---------------------------------------------------------------- 2. Markov chain: P(o_t | o_{t-1})

def fit_markov(train, seqs):
    """Count transitions 'level now -> level 5 minutes later' between neighbouring intervals
    that are both observed, add 1 to every count, and divide each row by its total.
    T[a, b] = P(next level = b | current level = a)."""
    counts = np.ones((K, K))
    for day in train:
        o = seqs[day]
        for now, nxt in zip(o[:-1], o[1:]):
            if now >= 0 and nxt >= 0:
                counts[now, nxt] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def predict_markov(T, p_start, levels):
    """For each observed interval, P(o_t | last observed level).
    If the last observed level was k intervals ago (missing intervals in between),
    use T multiplied k times, i.e. the probability of getting there in k steps.
    The first observed interval of the day has no previous level, so it uses p_start
    (the independent distribution), moved forward through any missing intervals before it."""
    preds = []
    prev, prev_t = None, None
    for t, o in enumerate(levels):
        if o < 0:
            continue
        if prev is None:
            preds.append(p_start @ np.linalg.matrix_power(T, t))
        else:
            preds.append(np.linalg.matrix_power(T, t - prev_t)[prev])
        prev, prev_t = o, t
    return np.array(preds).reshape(-1, K)


# ---------------------------------------------------------------- 3. time of day: P(o_t | hour)

def fit_time_of_day(train, seqs, starts, overall):
    """Count levels separately for every hour of the day. An hour with few intervals would
    get extreme probabilities, so each hour's counts get 2 extra 'pseudo-intervals' spread
    like the overall distribution (`overall` = the independent model) before dividing."""
    counts = {}
    for day in train:
        o = seqs[day]
        for hour, level in zip(hour_of_each_interval(day, len(o), starts), o):
            if level >= 0:
                counts.setdefault(hour, np.zeros(K))[level] += 1
    return {hour: (c + 2 * overall) / (c.sum() + 2) for hour, c in counts.items()}


def predict_time_of_day(table, overall, day, levels, starts):
    """Look up each observed interval's hour; an hour never seen in training uses `overall`."""
    hours = hour_of_each_interval(day, len(levels), starts)[levels >= 0]
    return np.array([table.get(h, overall) for h in hours]).reshape(-1, K)


# ---------------------------------------------------------------- all three at once

def fit_baselines(train, seqs, starts):
    """Fit the three baselines on the training days. Returns {name: predict(day, levels)}."""
    p = fit_independent(train, seqs)
    T = fit_markov(train, seqs)
    table = fit_time_of_day(train, seqs, starts, p)
    predictors = {
        "independent": lambda day, levels: predict_independent(p, levels),
        "markov": lambda day, levels: predict_markov(T, p, levels),
        "time_of_day": lambda day, levels: predict_time_of_day(table, p, day, levels, starts),
    }
    return predictors, {"independent": p, "markov": T, "time_of_day": table}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", nargs="+", default=ROUTES)
    ap.add_argument("--test-days", type=int, default=4)
    ap.add_argument("--obs", choices=OBSERVATIONS, default="level")
    args = ap.parse_args()

    for route in args.route:
        seqs, starts = load_route(route, args.obs)
        train, _ = split_days(seqs, args.test_days)
        _, tables = fit_baselines(train, seqs, starts)
        print(f"=== {route} ({args.obs}), learned from {train[0]}..{train[-1]} ({len(train)} days)")
        print("\nindependent: P(level)")
        print(pd.Series(tables["independent"], index=LEVELS).round(3).to_string())
        print("\nMarkov chain: P(next level | current level), rows = current level")
        print(pd.DataFrame(tables["markov"], index=LEVELS, columns=LEVELS).round(3).to_string())
        print("\ntime of day: P(level | hour)")
        print(pd.DataFrame(tables["time_of_day"], index=LEVELS).T.sort_index().round(3).to_string())
        print()


if __name__ == "__main__":
    main()
