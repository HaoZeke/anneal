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
    fig, ax = plt.subplots(figsize=(6, 3.5))
    width = 0.8 / len(series)
    for k, (label, path) in enumerate(series):
        z = coordination(read_xyz(path), 1.35 * length)
        counts = np.bincount(z, minlength=14)
        ax.bar(np.arange(14) + (k - (len(series) - 1) / 2) * width, counts, width, label=label)
    ax.set_xlabel("neighbours within 1.35 length scales")
    ax.set_ylabel("atoms")
    ax.set_xticks(range(3, 14))
    ax.set_xlim(2.5, 13.5)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(series))
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main()
