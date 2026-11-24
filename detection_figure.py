"""
Detection on Outside Loop, November 2 (a bus went missing in the afternoon), with the main
3-state HMM saved by hmm_train.py: (a) the observed excess-wait level, (b) the live belief
P(state | waits so far) from the forward algorithm, (c) the Viterbi path. Saves
figures/nov2_detection.png and prints the times the figure is about.

November 2 is a training day, so this shows what the model detects, not a held-out test.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import K
from data import STEP_MS, TZ, clock, hmm_path, load_route
from hmm import filter_beliefs, viterbi

FIGS_DIR = Path(__file__).parent / "figures"
ROUTE, DAY = "outside_loop", "2026-11-02"
STATES = ["normal", "delayed", "heavily delayed"]
LEVELS = ["normal or shorter", "longer", "much longer"]
COLORS = ["#8a8984", "#eda100", "#e34948"]      # gray, yellow, red: calm to disrupted
INK, MUTED = "#0b0b0b", "#52514e"

def runs(values, target):
    """(first, last) step of every run of consecutive steps equal to target."""
    out, t = [], 0
    while t < len(values):
        if values[t] == target:
            u = t
            while u + 1 < len(values) and values[u + 1] == target:
                u += 1
            out.append((t, u))
            t = u + 1
        else:
            t += 1
    return out

def main():
    seqs, starts = load_route(ROUTE)
    o, start = seqs[DAY], starts[DAY]
    saved = np.load(hmm_path(ROUTE, "level", K))
    pi, A, B = saved["pi"], saved["A"], saved["B"]
    beliefs = filter_beliefs(pi, A, B, o)
    path, _ = viterbi(pi, A, B, o)

    times = pd.to_datetime(start + np.arange(len(o)) * STEP_MS, unit="ms", utc=True).tz_convert(TZ).tz_localize(None)
    at = lambda t: clock(start + t * STEP_MS)
    print(f"{ROUTE} {DAY}: trained on {DAY in saved['train_days']} (a training day)")
    for level in (1, 2):
        print(f"  observed {LEVELS[level]}:", [(at(a), at(b)) for a, b in runs(o, level)])
    for state in (1, 2):
        print(f"  Viterbi {STATES[state]}:", [(at(a), at(b)) for a, b in runs(path, state)])
    first = np.argmax(beliefs[:, 2] > 0.5)
    print(f"  live belief in 'heavily delayed' first above 0.5 at {at(first)} (step {first}),"
          f" level that step = {o[first]}")

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.4, 1]})
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color="#e8e7e3", lw=0.6)

    # (a) observed level
    axes[0].step(times, o, where="post", color=INK, lw=1.2)
    axes[0].set_yticks([0, 1, 2], LEVELS)
    axes[0].set_ylim(-0.3, 2.3)
    axes[0].set_title("(a) Observed excess-wait level", fontsize=9, loc="left", color=INK)

    # (b) live belief: one line per state, direct labels at the right end plus a legend
    for k in range(3):
        axes[1].plot(times, beliefs[:, k], color=COLORS[k], lw=1.6, label=STATES[k])
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_yticks([0, 0.5, 1])
    axes[1].set_ylabel("probability", fontsize=8, color=MUTED)
    axes[1].set_title("(b) Live belief P(state | waits so far), forward algorithm",
                      fontsize=9, loc="left", color=INK)
    axes[1].legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.62))

    # (c) Viterbi path
    axes[2].step(times, path, where="post", color=INK, lw=1.2)
    axes[2].set_yticks([0, 1, 2], STATES)
    axes[2].set_ylim(-0.3, 2.3)
    axes[2].set_title("(c) Most likely state path, Viterbi (whole day)", fontsize=9, loc="left", color=INK)

    axes[2].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    axes[2].set_xlabel("Time of day, Outside Loop, November 2", fontsize=8, color=MUTED)
    fig.tight_layout(h_pad=0.6)

    FIGS_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGS_DIR / "nov2_detection.png", dpi=200, bbox_inches="tight")
    print(f"saved {FIGS_DIR / 'nov2_detection.png'}")

if __name__ == "__main__":
    main()
