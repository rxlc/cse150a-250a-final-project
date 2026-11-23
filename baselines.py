"""Three baseline models for the bus wait HMM.

Each baseline is a simpler model that the HMM has to beat. All three are learned by
counting and normalising, like the CPTs in class, with the same pseudo-count as the HMM
(config.PSEUDO = 0.1) added to every count so no level gets probability 0.

    independent   P(o_t)                 unigram: ignores time completely (does the past matter at all?)
    markov        P(o_t | o_{t-1})       bigram: uses the last level, no hidden states (do hidden states help?)
    time_of_day   P(o_t | hour of day)   one distribution per hour (is it just the time of day?)

Every predict_* function has the same interface as the HMM (see evaluate.py): for one day
and k steps ahead it returns a (T, M) array, row t = P(o_t | o_0 .. o_{t-k}), where M is
the observation's number of levels (config.OBSERVATIONS).
evaluate.score scores those predictions, the same way for every model.

Data: data.load_route, i.e. data.py's sequences for the kept weekdays, -1 = missing:
    level (core)        excess wait, levels 0 and 1 merged: 0 normal or shorter, 1 longer, 2 much longer
    ratio, delay        ablations, 4 levels each (see config.OBSERVATIONS and data.py)
The train/test split is data.split_days (test days in config.TEST_DAYS), shared with the HMM.

Run this file on its own to print the tables learned from the training days:
    python baselines.py
    python baselines.py --route mesa_loop --obs ratio
"""

import argparse

import numpy as np
import pandas as pd

from config import OBSERVATIONS, PSEUDO
from data import ROUTES, STEP_MS, TZ, load_route, split_days

LEVELS = {
    "level": ["normal_or_shorter", "longer", "much_longer"],
    "ratio": ["short", "regular", "long", "very_long"],
    "delay": ["early", "on_time", "late", "very_late"],
}


def hour_of_each_interval(day, n, starts):
    """Local hour of day (0-23) of each of the n intervals of a day."""
    t = pd.to_datetime(starts[day] + np.arange(n) * STEP_MS, unit="ms", utc=True)
    return t.tz_convert(TZ).hour.to_numpy()


# ---------------------------------------------------------------- 1. independent: P(o_t)

def fit_independent(train, seqs, M):
    """Count how often each level appears, add the pseudo-count to every count (so no level
    gets probability 0), and divide by the total."""
    counts = np.full(M, PSEUDO)
    for day in train:
        o = seqs[day]
        counts += np.bincount(o[o >= 0], minlength=M)
    return counts / counts.sum()


def predict_independent(p, levels, k):
    """The same distribution for every interval, whatever k is."""
    return np.tile(p, (len(levels), 1))


# ---------------------------------------------------------------- 2. Markov chain: P(o_t | o_{t-1})

def fit_markov(train, seqs, M):
    """Count transitions 'level now -> level 5 minutes later' between neighbouring intervals
    that are both observed, add the pseudo-count to every count, and divide each row by its
    total. T[a, b] = P(next level = b | current level = a)."""
    counts = np.full((M, M), PSEUDO)
    for day in train:
        o = seqs[day]
        for now, nxt in zip(o[:-1], o[1:]):
            if now >= 0 and nxt >= 0:
                counts[now, nxt] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def predict_markov(T, p_start, levels, k):
    """For each interval t, P(o_t | last observed level at or before t - k).
    If that level was seen at step s, use T multiplied t - s times (the probability of
    getting there in t - s steps; missing intervals in between just add steps).
    If nothing was observed yet, start from p_start (the independent distribution) at
    step 0 and move it forward t steps."""
    preds = np.zeros((len(levels), len(p_start)))
    for t in range(len(levels)):
        s = t - k
        while s >= 0 and levels[s] < 0:
            s -= 1
        if s < 0:
            preds[t] = p_start @ np.linalg.matrix_power(T, t)
        else:
            preds[t] = np.linalg.matrix_power(T, t - s)[levels[s]]
    return preds


# ---------------------------------------------------------------- 3. time of day: P(o_t | hour)

def fit_time_of_day(train, seqs, starts, M):
    """Count levels separately for every hour of the day, add the pseudo-count to every
    count, and divide each hour's counts by their total."""
    counts = {}
    for day in train:
        o = seqs[day]
        for hour, level in zip(hour_of_each_interval(day, len(o), starts), o):
            if level >= 0:
                counts.setdefault(hour, np.full(M, PSEUDO))[level] += 1
    return {hour: c / c.sum() for hour, c in counts.items()}


def predict_time_of_day(table, overall, day, levels, starts, k):
    """Look up each interval's hour, whatever k is; an hour never seen in training uses
    `overall` (the independent distribution)."""
    hours = hour_of_each_interval(day, len(levels), starts)
    return np.array([table.get(h, overall) for h in hours]).reshape(-1, len(overall))


# ---------------------------------------------------------------- all three at once

def fit_baselines(train, seqs, starts, M):
    """Fit the three baselines on the training days, for an observation with M levels.
    Returns {name: predict(day, levels, k)}."""
    p = fit_independent(train, seqs, M)
    T = fit_markov(train, seqs, M)
    table = fit_time_of_day(train, seqs, starts, M)
    predictors = {
        "independent": lambda day, levels, k: predict_independent(p, levels, k),
        "markov": lambda day, levels, k: predict_markov(T, p, levels, k),
        "time_of_day": lambda day, levels, k: predict_time_of_day(table, p, day, levels, starts, k),
    }
    return predictors, {"independent": p, "markov": T, "time_of_day": table}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", nargs="+", default=list(ROUTES.values()))
    ap.add_argument("--obs", choices=list(OBSERVATIONS), default="level")
    args = ap.parse_args()
    M, _ = OBSERVATIONS[args.obs]
    names = LEVELS[args.obs]

    for route in args.route:
        seqs, starts = load_route(route, args.obs)
        train, _ = split_days(seqs)
        _, tables = fit_baselines(train, seqs, starts, M)
        print(f"=== {route} ({args.obs}), learned from {train[0]}..{train[-1]} ({len(train)} days)")
        print("\nindependent: P(level)")
        print(pd.Series(tables["independent"], index=names).round(3).to_string())
        print("\nMarkov chain: P(next level | current level), rows = current level")
        print(pd.DataFrame(tables["markov"], index=names, columns=names).round(3).to_string())
        print("\ntime of day: P(level | hour)")
        print(pd.DataFrame(tables["time_of_day"], index=names).T.sort_index().round(3).to_string())
        print()


if __name__ == "__main__":
    main()
