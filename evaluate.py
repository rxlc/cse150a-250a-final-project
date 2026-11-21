"""Evaluation: is the HMM better than the baselines?

Every model learns from the training days only, then predicts the held-out test days one
5-minute interval at a time, and every model is scored the same way on those test days:

    test log-likelihood per interval   mean of log P(level that really happened | model)
                                       closer to 0 = the model predicted the real data better
                                       (this is the main score)
    accuracy                           how often the model's top guess was right; misleading
                                       here, since most intervals are "normal"

Models compared:
    independent, markov, time_of_day   the three baselines, from baselines.py
    the team's HMMs                    already trained, loaded from --hmm-dir (default ./hmms):
                                       one .npz per HMM with arrays pi, A, B, named
                                       <route>_<obs>_<name>.npz, e.g. outside_loop_level_hmm_3.npz
                                       IMPORTANT: they must be trained on the training days only
                                       (use split_days from baselines.py), or the test is not fair
This file does not train the HMM; it only scores it, with the forward algorithm.

Outputs, in ./results:
    results_<obs>.csv                  one row per route and model
    test_loglik_<obs>.png           test log-likelihood vs number of hidden states,
                                       with the baselines as horizontal lines

Usage (run `python data.py` first, it writes the sequences to output/):
    python evaluate.py                       # all three loops, `level` observation (core)
    python evaluate.py --route mesa_loop
    python evaluate.py --obs ratio           # ablations: --obs ratio, --obs delay
"""

import argparse

import numpy as np
import pandas as pd

from pathlib import Path

from baselines import K, OBSERVATIONS, RES_DIR, ROUTES, fit_baselines, load_route, split_days


def predict_hmm(pi, A, B, levels):
    """Forward algorithm, run as a prediction. Keeps a belief
    P(hidden state now | levels seen so far today), and for each observed interval returns
    P(level | levels seen so far) = belief @ B. Then it updates the belief with the level
    that really happened (Bayes' rule) and moves it one step forward through A.
    A missing interval (-1) gives no evidence, but the 5 minutes still pass.
    Normalising the belief every step keeps the numbers from underflowing on long days."""
    belief = pi.copy()
    preds = []
    for o in levels:
        if o >= 0:
            preds.append(belief @ B)          # predict, before seeing o
            belief = belief * B[:, o]         # update with what happened
            belief /= belief.sum()
        belief = belief @ A                   # 5 minutes pass
    return np.array(preds).reshape(-1, K)


def load_hmms(route, obs, hmm_dir):
    """{name: (pi, A, B)} for every <route>_<obs>_<name>.npz in hmm_dir."""
    hmms = {}
    prefix = f"{route}_{obs}_"
    for f in sorted(Path(hmm_dir).glob(f"{prefix}*.npz")):
        p = np.load(f)
        hmms[f.stem[len(prefix):]] = (p["pi"], p["A"], p["B"])
    return hmms


def score(predict, days, seqs):
    """Run one model over a set of days and grade its predictions."""
    P = np.vstack([predict(day, seqs[day]) for day in days])
    y = np.concatenate([seqs[day][seqs[day] >= 0] for day in days])
    return {
        "loglik_per_interval": np.mean(np.log(P[np.arange(len(y)), y])),
        "accuracy": np.mean(P.argmax(axis=1) == y),
        "n_intervals": len(y),
    }


def evaluate_route(route, obs, n_test, hmm_dir):
    seqs, starts = load_route(route, obs)
    train, test = split_days(seqs, n_test)
    print(f"=== {route} ({obs}): train {train[0]}..{train[-1]} ({len(train)} days), "
          f"test {test[0]}..{test[-1]} ({len(test)} days)")

    # 1. the baselines learn from the training days only; the HMMs were trained the same way
    predictors, _ = fit_baselines(train, seqs, starts)
    hmms = load_hmms(route, obs, hmm_dir)
    if not hmms:
        print(f"  (no HMMs found for {route} in {hmm_dir}/, scoring the baselines only)")
    n_states = {}
    for name, (pi, A, B) in hmms.items():
        predictors[name] = lambda day, levels, pi=pi, A=A, B=B: predict_hmm(pi, A, B, levels)
        n_states[name] = len(pi)

    # 2. every model is scored the same way, on the training days and on the test days
    rows = []
    for name, predict in predictors.items():
        tr, te = score(predict, train, seqs), score(predict, test, seqs)
        rows.append({"route": route, "obs": obs, "model": name,
                     "hidden_states": n_states.get(name),
                     "train_loglik": tr["loglik_per_interval"],
                     "test_loglik": te["loglik_per_interval"],
                     "test_accuracy": te["accuracy"],
                     "test_intervals": te["n_intervals"]})
    res = pd.DataFrame(rows)
    print(res.drop(columns=["route", "obs"]).to_string(index=False, float_format="%.3f"))
    print()
    return res


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    routes = list(res["route"].unique())
    fig, axes = plt.subplots(1, len(routes), figsize=(5 * len(routes), 4), squeeze=False)
    styles = {"independent": ":", "markov": "--", "time_of_day": "-."}
    for ax, route in zip(axes[0], routes):
        r = res[res["route"] == route]
        h = r[r["hidden_states"].notna()].sort_values("hidden_states")
        if len(h):
            ax.plot(h["hidden_states"], h["test_loglik"], "o-", color="black", label="HMM")
            ax.set_xticks(h["hidden_states"].astype(int).unique())
        for model, ls in styles.items():
            row = r[r["model"] == model]
            if len(row):
                ax.axhline(row["test_loglik"].iloc[0], ls=ls, color="gray", label=model)
        ax.set_xlabel("number of hidden states S")
        ax.set_ylabel("test log-likelihood per interval")
        ax.set_title(route)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"plot saved to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", nargs="+", default=ROUTES)
    ap.add_argument("--obs", choices=OBSERVATIONS, default="level")
    ap.add_argument("--test-days", type=int, default=4)
    ap.add_argument("--hmm-dir", default="hmms", help="folder with the trained HMMs (.npz)")
    args = ap.parse_args()

    res = pd.concat([evaluate_route(r, args.obs, args.test_days, args.hmm_dir)
                     for r in args.route])
    RES_DIR.mkdir(exist_ok=True)
    res.to_csv(RES_DIR / f"results_{args.obs}.csv", index=False)
    print(f"results saved to {RES_DIR / f'results_{args.obs}.csv'}")
    plot(res, RES_DIR / f"test_loglik_{args.obs}.png")


if __name__ == "__main__":
    main()
