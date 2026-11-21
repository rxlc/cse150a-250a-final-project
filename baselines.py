"""Three baseline models for the bus wait HMM.

Each baseline is a simpler model that the HMM has to beat. All three are learned by
maximum likelihood, i.e. by counting and normalising, like the CPTs in class.

    independent   P(o_t)                 ignores time completely (does the past matter at all?)
    markov        P(o_t | o_{t-1})       uses the last level, no hidden states (do hidden states help?)
    time_of_day   P(o_t | hour of day)   uses only the clock (is it just the time of day?)

Every predict_* function returns one predicted distribution over the 4 levels for each
observed interval of a day. evaluate.py scores those predictions.

Run this file on its own to print the learned tables:
    python baselines.py
    python baselines.py --route mesa_loop
"""

import argparse

import numpy as np
import pandas as pd

from load_data import K, LEVELS, ROUTES, hour_of_each_interval, load_route, split_days


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
    ap.add_argument("--route", nargs="+", default=list(ROUTES))
    ap.add_argument("--test-days", type=int, default=4)
    ap.add_argument("--source", choices=["intervals", "ratio"], default="intervals")
    args = ap.parse_args()

    for route in args.route:
        seqs, starts = load_route(route, args.source)
        train, _ = split_days(seqs, args.test_days)
        _, tables = fit_baselines(train, seqs, starts)
        print(f"=== {route}, learned from {train[0]}..{train[-1]} ({len(train)} days)")
        print("\nindependent: P(level)")
        print(pd.Series(tables["independent"], index=LEVELS).round(3).to_string())
        print("\nMarkov chain: P(next level | current level), rows = current level")
        print(pd.DataFrame(tables["markov"], index=LEVELS, columns=LEVELS).round(3).to_string())
        print("\ntime of day: P(level | hour)")
        print(pd.DataFrame(tables["time_of_day"], index=LEVELS).T.sort_index().round(3).to_string())
        print()


if __name__ == "__main__":
    main()
