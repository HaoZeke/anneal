# Copyright (c) anneal developers
"""The house style of the JCC figures, shared by every fig_*.py script that draws one.

A figure is a vector PDF at the JCC single-column or double-column width, set in a
Times-metric serif at 7 pt or more, drawn in the Ruhi palette of Ruhila Goswami as
chemparseplot ships it, with no title. The PDF carries no creation date, so a rerun of a
script on the same inputs writes the same bytes.

RAMP, the map of an energy or a rate, runs from teal through sky to sunshine, so its
lightness rises with its value; chemparseplot's ruhi_diverging passes through magenta,
darker than sky, so it does not.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt

SINGLE = 3.35
DOUBLE = 6.46
SMALLEST = 7.0
BASE = 8.0
CHEMPARSEPLOT = "1.11.0"


def _ruhi() -> dict[str, str]:
    from importlib.metadata import version

    if version("chemparseplot") != CHEMPARSEPLOT:
        raise SystemExit(f"chemparseplot {version('chemparseplot')} is not the {CHEMPARSEPLOT} whose palette the figures use")
    from chemparseplot.plot.theme import RUHI_COLORS

    return dict(RUHI_COLORS)


RUHI = _ruhi()
CORAL = RUHI["coral"]
SUNSHINE = RUHI["sunshine"]
TEAL = RUHI["teal"]
SKY = RUHI["sky"]
MAGENTA = RUHI["magenta"]
BLACK = "#000000"
GREY = "#999999"
PENDING = "TODO(verify): anneal_repro rule"


def ramp():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("ruhi_ramp", [TEAL, SKY, SUNSHINE], N=256)


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "Nimbus Roman", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": BASE,
            "axes.labelsize": BASE,
            "axes.titlesize": BASE,
            "xtick.labelsize": SMALLEST,
            "ytick.labelsize": SMALLEST,
            "legend.fontsize": SMALLEST,
            "legend.frameon": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "lines.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
            "hatch.linewidth": 0.5,
        }
    )


def pending(ax, x: float = 1.0, y: float = 1.02, **kwargs) -> None:
    """Mark a panel whose numbers wait on the anneal_repro rule, as the draft marks them."""
    ax.text(
        x, y, PENDING, transform=ax.transAxes, ha=kwargs.pop("ha", "right"), va=kwargs.pop("va", "bottom"),
        color="red", fontsize=SMALLEST, **kwargs,
    )


def label(ax, letter: str) -> None:
    ax.text(0.0, 1.02, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=BASE)


def plain(number: float) -> str:
    """A tick value as plain digits, thousands set off by a thin space, with no exponent."""
    text = f"{number:,.0f}".replace(",", "\N{THIN SPACE}") if abs(number) >= 1 else f"{number:g}"
    return text.replace("-", "\N{MINUS SIGN}")


def plain_log(axis) -> None:
    """Label the decades of a log axis as plain numbers, since an exponent would set below 7 pt."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    axis.set_major_formatter(FuncFormatter(lambda value, _: plain(value)))
    axis.set_minor_formatter(NullFormatter())


def canvas(fig, rect=(0.0, 0.0, 1.0, 1.0)):
    """An axes over rect of fig whose data units are inches of the figure, for a schematic."""
    width, height = fig.get_size_inches()
    ax = fig.add_axes(rect)
    ax.set_xlim(0, width * rect[2])
    ax.set_ylim(0, height * rect[3])
    ax.set_axis_off()
    return ax


def box(ax, x, y, w, h, text="", fill="white", edge=BLACK, size=SMALLEST, weight="normal", **kwargs):
    """A rounded box with its lower left at (x, y), and its text centred in it."""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", facecolor=fill, edgecolor=edge,
            linewidth=0.7, **kwargs,
        )
    )
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, linespacing=1.15)
    return x, y, w, h


def arrow(ax, start, end, colour=BLACK, both=False, rad=0.0, **kwargs) -> None:
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="<|-|>" if both else "-|>", mutation_scale=7, color=colour,
            linewidth=0.7, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}", **kwargs,
        )
    )


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", metadata={"CreationDate": None, "ModDate": None, "Creator": None})
    plt.close(fig)
