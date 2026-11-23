"""Evaluation: is the HMM better than the baselines?

Every model learns from the training days only (all days except config.TEST_DAYS, the same
for all models), then predicts the held-out test days, and every model is scored by score() below.

Every model gives, for one day and a number of steps ahead k, a (T, M) array `preds`:
    preds[t] = P(o_t | o_0 .. o_{t-k})     predicted distribution of the level at step t,
                                           made k steps (k * 5 minutes) earlier
With k = 1 this is the usual one-step prediction used for the log-likelihood.

    test log-likelihood per interval   mean of log P(level that really happened | levels seen so far)
                                       closer to 0 = the model predicted the real data better
                                       (this is the main score)
    Brier score at k = 1, 2, 6         sum over levels of (predicted p - 1 if it happened else 0)^2,
                                       for the prediction made k steps (5, 10, 30 min) earlier;
                                       0 is perfect, lower is better

Accuracy is not reported: most intervals are "normal", so always guessing "normal" scores well.
Scores of different observations (--obs) are not comparable with each other: compare each
observation's HMM only with the baselines on the same observation.

Models compared:
    independent, markov, time_of_day   the three baselines, from baselines.py (fitted here by
                                       counting on the training days)
    hmm_K3                             the main HMM (config.K = 3 states) that hmm_train.py saved
                                       in output/<route>_<obs>_K3_hmm.npz (trained without the
                                       test days); this file does not train the HMM
    hmm_K2, hmm_K4, ...                with --k-ablation: an HMM for every K in config.K_CANDIDATES,
                                       all reported on the test days. This is an ablation, not a
                                       choice: the main model stays K = 3 whatever these scores are

Outputs, in ./results:
    results_<obs>.csv                  one row per route and model, all test days together
    results_<obs>_by_day.csv           one row per route, model and test day

Usage (run `python data.py`, then `python hmm_train.py --obs ...` first):
    python evaluate.py                       # core observation, all three loops
    python evaluate.py --obs ratio           # ablations: --obs ratio, --obs delay
    python evaluate.py --k-ablation          # K = 2, 3, 4 (run hmm_train.py --k-ablation first)
    python evaluate.py --route mesa_loop
"""

import argparse

import numpy as np
import pandas as pd

from baselines import fit_baselines
from config import K, K_CANDIDATES, OBSERVATIONS, STEPS_AHEAD
from data import OUT_DIR, ROUTES, load_route, split_days
from hmm import forecasts

RES_DIR = OUT_DIR.parent / "results"


# ---------------------------------------------------------------- metrics, the same for every model

def log_lik(preds, levels):
    """Sum of log P(observed level) over the observed steps, and how many there were.
    preds must be the k = 1 predictions."""
    total, n = 0.0, 0
    for t, o in enumerate(levels):
        if o >= 0:
            total += np.log(preds[t, o])
            n += 1
    return total, n


def brier(preds, levels, k):
    """Sum of Brier scores, sum over levels of (p_m - [o_t = m])^2, over the observed steps
    t >= k (steps that have k earlier steps to predict from), and how many there were."""
    total, n = 0.0, 0
    for t in range(k, len(levels)):
        o = levels[t]
        if o >= 0:
            truth = np.zeros(preds.shape[1])
            truth[o] = 1
            total += np.sum((preds[t] - truth) ** 2)
            n += 1
    return total, n


def score(predict, days, seqs):
    """Score one model on a set of days. predict(day, levels, k) -> (T, M) array as above.
    Returns log-likelihood per observed interval and the mean Brier score at each k."""
    ll, n_ll = 0.0, 0
    br = {k: [0.0, 0] for k in STEPS_AHEAD}
    for day in days:
        levels = seqs[day]
        s, n = log_lik(predict(day, levels, 1), levels)
        ll, n_ll = ll + s, n_ll + n
        for k in STEPS_AHEAD:
            s, n = brier(predict(day, levels, k), levels, k)
            br[k][0] += s
            br[k][1] += n
    result = {"loglik_per_interval": ll / n_ll, "n_intervals": n_ll}
    for k in STEPS_AHEAD:
        result[f"brier_k{k}"] = br[k][0] / br[k][1]
    return result


# ---------------------------------------------------------------- the HMM, as a predictor

def predict_hmm(pi, A, B, levels, k):
    """Same interface as the baselines: row t = P(o_t | o_0 .. o_{t-k}).
    For t >= k this is hmm.forecasts (filtered belief at t - k, moved k steps through A, then B).
    For t < k nothing has been seen yet, so it is pi moved t steps through A, then B."""
    preds = np.zeros((len(levels), B.shape[1]))
    for t in range(min(k, len(levels))):
        preds[t] = pi @ np.linalg.matrix_power(A, t) @ B
    preds[k:] = forecasts(pi, A, B, levels, k)
    return preds


def load_hmm(route, obs, n_states, train, test):
    """predict(day, levels, k) for the HMM with n_states hidden states in
    output/<route>_<obs>_K<n_states>_hmm.npz. Checks it was trained on the current training
    days and not on the test days, so a stale file is not used."""
    path = OUT_DIR / f"{route}_{obs}_K{n_states}_hmm.npz"
    saved = np.load(path)
    if ("train_days" not in saved or list(saved["train_days"]) != list(train)
            or list(saved["test_days"]) != list(test)):
        raise RuntimeError(f"{path.name} was not trained on the current training days; "
                           f"rerun hmm_train.py --obs {obs}")
    pi, A, B = saved["pi"], saved["A"], saved["B"]
    return lambda day, levels, k: predict_hmm(pi, A, B, levels, k)


def evaluate_route(route, obs, all_states):
    M, _ = OBSERVATIONS[obs]
    seqs, starts = load_route(route, obs)

    train, test = split_days(seqs)
    print(f"=== {route} ({obs}, M = {M}): train {len(train)} days, test {', '.join(test)}")

    # 1. every model comes from the training days only
    predictors, _ = fit_baselines(train, seqs, starts, M)
    for n_states in all_states:
        predictors[f"hmm_K{n_states}"] = load_hmm(route, obs, n_states, train, test)

    # 2. every model is scored the same way: on all test days together, and on each test day
    rows, day_rows = [], []
    for name, predict in predictors.items():
        rows.append({"route": route, "obs": obs, "model": name, **score(predict, test, seqs)})
        for day in test:
            day_rows.append({"route": route, "obs": obs, "model": name, "day": day,
                             **score(predict, [day], seqs)})

    res, by_day = pd.DataFrame(rows), pd.DataFrame(day_rows)
    print(res.drop(columns=["route", "obs"]).to_string(index=False, float_format="%.4f"))

    # per test day: each HMM next to the Markov chain, and on how many days the HMM was better
    cols = ["loglik_per_interval"] + [f"brier_k{k}" for k in STEPS_AHEAD]
    markov_d = by_day[by_day["model"] == "markov"].set_index("day")[cols]
    for n_states in all_states:
        name = f"hmm_K{n_states}"
        hmm_d = by_day[by_day["model"] == name].set_index("day")[cols]
        print(f"\nper test day, {name} vs markov (log-lik: higher is better, Brier: lower is better)")
        print(pd.concat({name: hmm_d, "markov": markov_d}, axis=1).to_string(float_format="%.4f"))
        wins = [f"log-lik {(hmm_d['loglik_per_interval'] > markov_d['loglik_per_interval']).sum()}"]
        for k in STEPS_AHEAD:
            wins.append(f"Brier k={k} {(hmm_d[f'brier_k{k}'] < markov_d[f'brier_k{k}']).sum()}")
        print(f"days (of {len(test)}) {name} beat the Markov chain: " + ", ".join(wins))
    print()
    return res, by_day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", nargs="+", default=list(ROUTES.values()))
    ap.add_argument("--obs", choices=list(OBSERVATIONS), default="level")
    ap.add_argument("--k-ablation", action="store_true", help="also score every K in config.K_CANDIDATES")
    args = ap.parse_args()
    all_states = K_CANDIDATES if args.k_ablation else [K]

    results = [evaluate_route(r, args.obs, all_states) for r in args.route]
    RES_DIR.mkdir(exist_ok=True)
    pd.concat([res for res, _ in results]).to_csv(RES_DIR / f"results_{args.obs}.csv", index=False)
    pd.concat([by_day for _, by_day in results]).to_csv(RES_DIR / f"results_{args.obs}_by_day.csv", index=False)
    print(f"results saved to {RES_DIR / f'results_{args.obs}.csv'} and results_{args.obs}_by_day.csv")


if __name__ == "__main__":
    main()
