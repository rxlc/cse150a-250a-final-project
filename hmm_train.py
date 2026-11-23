"""Fit one HMM per route for one observation, on the training days only (all days except
config.TEST_DAYS), and save it in output/<route>_<obs>_K<number of states>_hmm.npz:

    pi, A, B, log_p        the HMM, trained on the training days
    train_days, test_days  the days it was trained on, and the test days it never saw;
                           evaluate.py checks both, then scores it on the test days

The main model has config.K = 3 hidden states. With --k-ablation, it also trains an HMM for
every number of states in config.K_CANDIDATES; evaluate.py --k-ablation reports all of them
on the test days. The main model stays K = 3 whatever those scores are.

    python hmm_train.py                       # core observation (excess wait level), K = 3
    python hmm_train.py --k-ablation          # K = 2, 3, 4
    python hmm_train.py --obs ratio           # ablations: --obs ratio, --obs delay
"""

import argparse

import numpy as np

from config import K, K_CANDIDATES, OBSERVATIONS, PSEUDO, SEED
from data import OUT_DIR, ROUTES, load_route, split_days
from hmm import baum_welch

np.set_printoptions(precision=3, suppress=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", choices=list(OBSERVATIONS), default="level")
    ap.add_argument("--k-ablation", action="store_true", help="train every K in config.K_CANDIDATES")
    args = ap.parse_args()
    M, _ = OBSERVATIONS[args.obs]
    all_states = K_CANDIDATES if args.k_ablation else [K]

    for name in ROUTES.values():
        seqs, _ = load_route(name, args.obs)
        train, test = split_days(seqs)
        days = [seqs[d] for d in train]
        n_steps = sum(len(o) for o in days)

        for n_states in all_states:
            print(f"{name} ({args.obs}, M = {M}, K = {n_states}): train {len(train)} days, test {', '.join(test)}")
            pi, A, B, log_p = baum_welch(days, K=n_states, M=M, pseudo=PSEUDO, seed=SEED)
            np.savez(OUT_DIR / f"{name}_{args.obs}_K{n_states}_hmm.npz", pi=pi, A=A, B=B, log_p=log_p,
                     train_days=np.array(train), test_days=np.array(test))

            print(f" best log P = {log_p:.2f}  ({log_p / n_steps:.3f} per step)")
            print("  pi =", pi)
            print("  A (row = from, column = to) =\n", A)
            print(f"  B (row = state, column = level 0-{M - 1}) =\n", B)
            print("  expected level per state =", B @ np.arange(B.shape[1]))
            print("  average time in each state (minutes) =", 5 / (1 - np.diag(A)))
            print()

if __name__ == "__main__":
    main()
