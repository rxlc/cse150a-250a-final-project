"""Fit one HMM per route on all weekdays and save the learned tables"""

import numpy as np
 
from data import OUT_DIR, ROUTES, load_sequences, merge_0_1_levels
from hmm import baum_welch
 
np.set_printoptions(precision=3, suppress=True)

def main():
    for name in ROUTES.values():
        days = [merge_0_1_levels(o) for o in load_sequences(name, "level").values()]
        print(f"{name}: {len(days)} days")
        pi, A, B, log_p = baum_welch(days)
        np.savez(OUT_DIR / f"{name}_hmm.npz", pi=pi, A=A, B=B, log_p=log_p)
 
        n_steps = sum(len(o) for o in days)

        print(f" best log P = {log_p:.2f}  ({log_p / n_steps:.3f} per step)")
        print("  pi =", pi)
        print("  A (row = from, column = to) =\n", A)
        print("  B (row = state, column = level 0-3) =\n", B)
        print("  expected level per state =", B @ np.arange(B.shape[1]))
        print("  average time in each state (minutes) =", 5 / (1 - np.diag(A)))
        print()
 
if __name__ == "__main__":
    main()
