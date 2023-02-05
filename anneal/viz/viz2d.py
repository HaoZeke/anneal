import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from anneal.funcs.obj2d import *
from anneal.core.components import AcceptStates, Quencher


class Plot2dObj:
    """Meant to plot 2D objects"""

    def __init__(self, obj: ObjectiveFunction, nelem: int):
        self.func = obj
        self.nelem = nelem
        step_size_x = (
            abs(self.func.limits.low[0] - self.func.limits.high[0]) / nelem
        )
        print(step_size_x)
        step_size_y = (
            abs(self.func.limits.low[1] - self.func.limits.high[1]) / nelem
        )
        print(step_size_y)
        plX = np.arange(
            self.func.limits.low[0], self.func.limits.high[0], step_size_x
        )
        plY = np.arange(
            self.func.limits.low[1], self.func.limits.high[1], step_size_y
        )
        self.X, self.Y = np.meshgrid(plX, plY, indexing="xy")
        self.Z = self.prepVals()
        self.contourExtent = [
            np.min(plX),
            np.max(plX),
            np.min(plY),
            np.max(plY),
        ]
        self.X_glob_min = self.X.ravel()[self.Z.argmin()]
        self.Y_glob_min = self.Y.ravel()[self.Z.argmin()]
        self.Z_glob_min = np.min(self.Z.ravel())
        self.pdat = None

    def prepVals(self):
        grid_vals = [
            self.func(np.column_stack([self.X[itera], self.Y[itera]]))
            for itera in range(self.nelem)
        ]
        return np.array(grid_vals)

    def create3d(self, showGlob=True, savePath=None):
        fig = plt.figure(figsize=(12, 10))
        ax = plt.subplot(projection="3d")
        surf = ax.plot_surface(
            self.X, self.Y, self.Z, cmap=cm.coolwarm, alpha=0.7
        )
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        [t.set_va("center") for t in ax.get_yticklabels()]
        [t.set_ha("left") for t in ax.get_yticklabels()]
        [t.set_va("center") for t in ax.get_xticklabels()]
        [t.set_ha("right") for t in ax.get_xticklabels()]
        [t.set_va("center") for t in ax.get_zticklabels()]
        [t.set_ha("left") for t in ax.get_zticklabels()]
        fig.colorbar(surf, shrink=0.35, aspect=8)
        ax.view_init(elev=15, azim=220)
        if showGlob:
            ax.scatter(
                self.X_glob_min,
                self.Y_glob_min,
                self.Z_glob_min,
                color="black",
                alpha=1,
            )
            ax.text(
                self.X_glob_min,
                self.Y_glob_min,
                self.Z_glob_min,
                "Global Minima",
                color="black",
                alpha=1,
            )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(str(self.func))
        if savePath is not None:
            plt.savefig(savePath, dpi=300)
        else:
            plt.show()

    def createContour(self, showGlob=True, savePath=None):
        fig = plt.figure(figsize=(12, 10))
        ax = plt.subplot()
        [t.set_va("center") for t in ax.get_yticklabels()]
        [t.set_ha("left") for t in ax.get_yticklabels()]
        [t.set_va("center") for t in ax.get_xticklabels()]
        [t.set_ha("right") for t in ax.get_xticklabels()]
        plt.imshow(
            self.Z,
            extent=[
                np.min(self.X.ravel()),
                np.max(self.X.ravel()),
                np.min(self.Y.ravel()),
                np.max(self.Y.ravel()),
            ],
            origin="lower",
            cmap="viridis",
            alpha=0.8,
        )
        plt.colorbar()
        contours = ax.contour(
            self.X, self.Y, self.Z, 10, colors="black", alpha=0.9
        )
        plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
        if showGlob:
            plt.plot(
                self.X_glob_min,
                self.Y_glob_min,
                color="white",
                marker="x",
                markersize=5,
            )
            ax.text(
                self.X_glob_min + 0.1,
                self.Y_glob_min,
                "Global Minima",
                color="white",
            )
        if savePath is not None:
            plt.savefig(savePath, dpi=300)
        else:
            plt.show()

    def plotQuenchContour(self, quenchy: Quencher, nsamples=500, savePath=None):
        self.pdat = pd.DataFrame(quenchy.PlotData)
        accepted_pos = self.pdat[self.pdat.accept == AcceptStates.IMPROVED][
            "pos"
        ]
        rejected_pos = self.pdat[self.pdat.accept == AcceptStates.REJECT]["pos"]
        mhaccept_pos = self.pdat[self.pdat.accept == AcceptStates.MHACCEPT][
            "pos"
        ]
        getDat = (
            lambda posDat: pd.DataFrame.sample(posDat, n=nsamples)
            if len(posDat) > nsamples
            else posDat
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
            extent=[
                np.min(self.X.ravel()),
                np.max(self.X.ravel()),
                np.min(self.Y.ravel()),
                np.max(self.Y.ravel()),
            ],
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
        ax.plot(
            quenchy.best.pos[0],
            quenchy.best.pos[1],
            marker="*",
            color="white",
            alpha=1,
        )
        for point in getDat(accepted_pos):
            ax.plot(point[0], point[1], marker="o", color="blue", alpha=0.5)
        for point in getDat(rejected_pos):
            plotReject(point)
        for point in getDat(mhaccept_pos):
            plotMHAccept(point)
        if savePath is not None:
            plt.savefig(savePath, dpi=300)
        else:
            plt.show()
