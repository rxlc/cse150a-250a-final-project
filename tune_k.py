"""Parameter testing for the number of hidden states K, on the training days only.

Choosing K by its score on the test days would tune the model to them, and the test score
would no longer be a fair estimate. So the last N_VALIDATION training days (just before the
test days) act as validation days: for each K in config.K_CANDIDATES, fit the HMM on the
other training days and score it on the validation days with evaluate.score. The Markov
chain, fitted on the same days, is printed as a reference. The test days are never used.
Pick the K with the best validation log-likelihood, set K in config.py, and rerun hmm_train.py.

    python tune_k.py                          # level, all three loops
    python tune_k.py --route mesa_loop
"""

import argparse

from baselines import fit_baselines
from config import K_CANDIDATES, OBSERVATIONS, PSEUDO, SEED
from data import ROUTES, load_route, split_days
from evaluate import predict_hmm, score
from hmm import baum_welch

N_VALIDATION = 3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", nargs="+", default=list(ROUTES.values()))
    ap.add_argument("--obs", choices=list(OBSERVATIONS), default="level")
    args = ap.parse_args()
    M, _ = OBSERVATIONS[args.obs]

    for route in args.route:
        seqs, starts = load_route(route, args.obs)
        train, _ = split_days(seqs)
        fit_days, valid = train[:-N_VALIDATION], train[-N_VALIDATION:]
        print(f"=== {route} ({args.obs}): fit {len(fit_days)} days, validate on {', '.join(valid)}")

        rows = []
        predictors, _ = fit_baselines(fit_days, seqs, starts, M)
        rows.append(("markov", score(predictors["markov"], valid, seqs)))
        for K in K_CANDIDATES:
            pi, A, B, _ = baum_welch([seqs[d] for d in fit_days], K=K, M=M, pseudo=PSEUDO, seed=SEED)
            predict = lambda day, levels, k: predict_hmm(pi, A, B, levels, k)
            rows.append((f"hmm K={K}", score(predict, valid, seqs)))

        for name, s in rows:
            print(f"  {name:10s} log-lik {s['loglik_per_interval']:.4f}  "
                  + "  ".join(f"{key} {v:.4f}" for key, v in s.items() if key.startswith("brier")))
        best = max(rows[1:], key=lambda r: r[1]["loglik_per_interval"])[0]
        print(f"  best HMM on validation: {best}\n")

if __name__ == "__main__":
    main()
