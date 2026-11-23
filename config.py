K = 3                      # hidden states of the HMM
SEED = 0                   # random restarts of Baum-Welch
PSEUDO = 0.1               # pseudo-count for the HMM and baselines
STEPS_AHEAD = [1, 2, 6]    # Brier score at 5, 10 and 30 minutes ahead

TEST_DAYS = ["2026-11-17", "2026-11-19", "2026-11-20"]

K_CANDIDATES = [2, 3, 4]

OBSERVATIONS = {
    "level": (3, True),
    "ratio": (4, False),
    "delay": (4, False),
}
