"""Fit one HMM per route for one observation, on the training days only (all days except
config.TEST_DAYS), and save it in output/<route>_<obs>_hmm.npz:

    pi, A, B, log_p        the HMM, trained on the training days
    train_days, test_days  the days it was trained on, and the test days it never saw;
                           evaluate.py checks both, then scores it on the test days

    python hmm_train.py                 # core observation (excess wait level)
    python hmm_train.py --obs ratio     # ablations: --obs ratio, --obs delay
"""

import argparse

import numpy as np

from config import K, OBSERVATIONS, PSEUDO, SEED
from data import OUT_DIR, ROUTES, load_route, split_days
from hmm import baum_welch

np.set_printoptions(precision=3, suppress=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", choices=list(OBSERVATIONS), default="level")
    args = ap.parse_args()
    M, _ = OBSERVATIONS[args.obs]

    for name in ROUTES.values():
        seqs, _ = load_route(name, args.obs)
        train, test = split_days(seqs)
        print(f"{name} ({args.obs}, M = {M}): train {len(train)} days, test {', '.join(test)}")

        days = [seqs[d] for d in train]
        pi, A, B, log_p = baum_welch(days, K=K, M=M, pseudo=PSEUDO, seed=SEED)
        np.savez(OUT_DIR / f"{name}_{args.obs}_hmm.npz", pi=pi, A=A, B=B, log_p=log_p,
                 train_days=np.array(train), test_days=np.array(test))

        n_steps = sum(len(o) for o in days)

        print(f" best log P = {log_p:.2f}  ({log_p / n_steps:.3f} per step)")
        print("  pi =", pi)
        print("  A (row = from, column = to) =\n", A)
        print(f"  B (row = state, column = level 0-{M - 1}) =\n", B)
        print("  expected level per state =", B @ np.arange(B.shape[1]))
        print("  average time in each state (minutes) =", 5 / (1 - np.diag(A)))
        print()

if __name__ == "__main__":
    main()
