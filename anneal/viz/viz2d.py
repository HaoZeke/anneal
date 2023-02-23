import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from anneal.core.components import AcceptStates, Quencher
from eindir.core.components import ObjectiveFunction
from eindir.viz.viz2d import Plot2dObj


class Plot2dQuench(Plot2dObj):
    """Meant to plot 2D Quench objects"""

    def __init__(self, obj: ObjectiveFunction, nelem: int):
        super().__init__(obj, nelem)

    def plotQuenchContour(
        self, quenchy: Quencher, nsamples=500, savePath=None, ptitle=None
    ):
        self.pdat = pd.DataFrame(quenchy.PlotData)
        accepted_pos = self.pdat[self.pdat.accept == AcceptStates.IMPROVED][
            "pos"
        ]
        rejected_pos = self.pdat[self.pdat.accept == AcceptStates.REJECT]["pos"]
        mhaccept_pos = self.pdat[self.pdat.accept == AcceptStates.MHACCEPT][
            "pos"
        ]
        toNPD = (
            lambda poscol: np.concatenate(poscol.to_list(), axis=0).reshape(
                -1, self.func.limits.dims
            )
            if len(poscol) > 1
            else poscol
        )
        getDat = (
            lambda posDat: toNPD(pd.DataFrame.sample(posDat, n=nsamples))
            if len(posDat) > nsamples
            else toNPD(posDat)
        )
        inBounds = lambda point: np.all(
            point > self.contourExtent[::2]
        ) and np.all(point < self.contourExtent[1::2])
        fig = plt.figure(figsize=(12, 10))
        ax = plt.subplot()
        plotAccept = (
            lambda point: ax.plot(
                point[0], point[1], marker="o", color="blue", alpha=0.5
            )
            if inBounds(point)
            else None
        )
        plotReject = (
            lambda point: ax.plot(
                point[0], point[1], marker="o", color="red", alpha=0.3
            )
            if inBounds(point)
            else None
        )
        plotMHAccept = (
            lambda point: ax.plot(
                point[0],
                point[1],
                marker="*",
                color="yellow",
                alpha=0.5,
            )
            if inBounds(point)
            else None
        )
        [t.set_va("center") for t in ax.get_yticklabels()]
        [t.set_ha("left") for t in ax.get_yticklabels()]
        [t.set_va("center") for t in ax.get_xticklabels()]
        [t.set_ha("right") for t in ax.get_xticklabels()]
        plt.imshow(
            self.Z,
            extent=self.contourExtent,
            origin="lower",
            cmap="viridis",
            alpha=0.8,
        )
        plt.colorbar()
        contours = ax.contour(
            self.X, self.Y, self.Z, 10, colors="black", alpha=0.9
        )
        plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
        plt.plot(
            self.X_glob_min,
            self.Y_glob_min,
            color="white",
            marker="x",
            markersize=5,
        )
        if not isinstance(ptitle, type(None)):
            plt.title(ptitle)
        if inBounds(quenchy.best.pos):
            ax.plot(
                quenchy.best.pos[0],
                quenchy.best.pos[1],
                marker="*",
                color="white",
                alpha=1,
            )
        else:
            ax.plot(
                quenchy.best.pos[0],
                quenchy.best.pos[1],
                marker="*",
                color="black",
                alpha=1,
            )
        for point in getDat(accepted_pos):
            plotAccept(point)
        for point in getDat(rejected_pos):
            plotReject(point)
        for point in getDat(mhaccept_pos):
            plotMHAccept(point)
        if savePath is not None:
            plt.savefig(savePath, dpi=300)
        else:
            plt.show()
