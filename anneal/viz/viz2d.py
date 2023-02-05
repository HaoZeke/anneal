import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from anneal.funcs.obj2d import *


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

    def prepVals(self):
        grid_vals = [
            self.func(np.column_stack([self.X[itera], self.Y[itera]]))
            for itera in range(self.nelem)
        ]
        return np.array(grid_vals)

    def create3d(self, showGlob=False, savePath=None):
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
                self.func.glob_minma_pos[0],
                self.func.glob_minma_pos[1],
                self.func.glob_minma_val,
                color="black",
                alpha=1,
            )
            ax.text(
                self.func.glob_minma_pos[0],
                self.func.glob_minma_pos[1],
                self.func.glob_minma_val,
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

    def createContour(self, showGlob=False, savePath=None):
        fig = plt.figure(figsize=(12, 10))
        ax = plt.subplot()
        [t.set_va("center") for t in ax.get_yticklabels()]
        [t.set_ha("left") for t in ax.get_yticklabels()]
        [t.set_va("center") for t in ax.get_xticklabels()]
        [t.set_ha("right") for t in ax.get_xticklabels()]
        ax.contourf(self.X, self.Y, self.Z)
        if showGlob:
            plt.scatter(
                self.func.glob_minma_pos[0],
                self.func.glob_minma_pos[1],
                color="black",
            )
            ax.text(
                self.func.glob_minma_pos[0],
                self.func.glob_minma_pos[1],
                "Global Minima",
            )
        plt.colorbar()
        if savePath is not None:
            plt.savefig(savePath, dpi=300)
        else:
            plt.show()
