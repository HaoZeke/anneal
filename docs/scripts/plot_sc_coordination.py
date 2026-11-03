"""Coordination histograms of Sutton-Chen minima written by sc_minima.

usage: plot_sc_coordination.py <length> <out.png> <label=file.xyz>...

Neighbours are pairs closer than 1.35 times `length`, the cutoff the hop
and sc_minima use.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

#: Neighbour cutoff in length scales, the library's `CONTACT_CUTOFF`.
CONTACT_CUTOFF = 1.35
#: Coordination bins plotted, from zero neighbours up.
BINS = 14
#: Lowest coordination that gets a tick label.
FIRST_TICK = 3
#: Half a bin, the margin beside the outer bars.
HALF_BIN = 0.5
#: Width that the bars of one coordination share.
GROUP_WIDTH = 0.8
#: Figure size in inches.
FIGSIZE = (6, 3.5)
#: Resolution of the saved figure in dots per inch.
DPI = 150
#: Legend anchor, centred above the plot.
LEGEND_ANCHOR = (0.5, 1.0)


def read_xyz(path):
    lines = open(path).read().splitlines()
    n = int(lines[0])
    return np.array([[float(v) for v in line.split()[1:4]] for line in lines[2 : 2 + n]])


def coordination(x, cutoff):
    d = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
    return ((d < cutoff) & (d > 0)).sum(1)


def main():
    length, out = float(sys.argv[1]), sys.argv[2]
    series = [arg.split("=", 1) for arg in sys.argv[3:]]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    width = GROUP_WIDTH / len(series)
    for k, (label, path) in enumerate(series):
        z = coordination(read_xyz(path), CONTACT_CUTOFF * length)
        counts = np.bincount(z, minlength=BINS)
        ax.bar(np.arange(BINS) + (k - (len(series) - 1) / 2) * width, counts, width, label=label)
    ax.set_xlabel("neighbours within 1.35 length scales")
    ax.set_ylabel("atoms")
    ax.set_xticks(range(FIRST_TICK, BINS))
    ax.set_xlim(FIRST_TICK - HALF_BIN, BINS - HALF_BIN)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=LEGEND_ANCHOR, ncol=len(series))
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)


if __name__ == "__main__":
    main()
