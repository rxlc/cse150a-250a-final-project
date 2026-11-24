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
    "level4": (4, False),  # excess wait without merging levels 0 and 1 (learns the timetable, Section 4.2)
}

# yes/no delay forecast: "delayed" means level >= DELAYED_FROM[obs]
# (level: longer or much longer; ratio: long or very long gap; delay: late or very late)
DELAYED_FROM = {"level": 1, "ratio": 2, "delay": 2, "level4": 2}
