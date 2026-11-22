"""
Based on data derived in data.py, builds a histogram of raw excess wait with the level cut-offs
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator, StrMethodFormatter

from data import ROUTES, keep_days, load, local_day

FIGS_DIR = Path(__file__).parent / "figures"

NAMES = {"inside_loop": "Inside Loop", "outside_loop": "Outside Loop", "mesa_loop": "Mesa Loop"}
ROUTE_COLORS = {"inside_loop": "#c49112", "outside_loop": "#03629C", "mesa_loop": "#1baf7a"}
CUTS = [-60, 120, 300]

def main(lo=-300, hi=720, width=30):
    iv = load("intervals_5min").dropna(subset=["level", "excess_wait_s"]).copy()
    iv["day"] = local_day(iv["slot_start"])
    iv = keep_days(iv)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    bounds = [lo, *CUTS, hi]
    for k in range(4):
        ax.text((bounds[k] + bounds[k + 1]) / 120, 1.01, str(k), transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8)

    ax.text(0.5, 1.10, "Observation level", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=8)

    for c in CUTS:
        ax.axvline(c / 60, color="#8a8984", lw=0.8, ls="--", zorder=1)

    edges = np.arange(lo, hi + width, width)
    for route_id, name in ROUTES.items():
        x = iv.loc[iv["route_id"] == route_id, "excess_wait_s"].to_numpy()
        counts, _ = np.histogram(np.clip(x, lo, hi - 1e-9), edges)
        ax.stairs(counts, edges / 60, color=ROUTE_COLORS[name], lw=1.5, label=NAMES[name], zorder=2)
        print(f"{name}: {np.sum(x < lo)} below {lo} s, {np.sum(x >= hi)} at or above {hi} s")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(lo / 60, hi / 60)
    ax.set_ylim(top=1100)
    ax.set_xlabel("Excess wait (minutes)")
    ax.set_ylabel("Intervals (log scale)")
    
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()

    FIGS_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGS_DIR / "excess_wait_hist.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()