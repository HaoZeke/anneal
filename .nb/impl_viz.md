---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: []

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
```

```{code-cell} ipython3
:tags: []

from anneal.funcs.obj2d import *
from anneal.funcs.objNd import *
```

```{code-cell} ipython3
:tags: []

class Plot2dObj():
    """Meant to plot 2D objects"""
    def __init__(self, obj: ObjectiveFunction, nelem: int):
        self.func = obj
        self.nelem = nelem
        step_size_x = abs(self.func.limits.low[0] - self.func.limits.high[0]) / nelem
        print(step_size_x)
        step_size_y = abs(self.func.limits.low[1] - self.func.limits.high[1]) / nelem
        print(step_size_y)
        plX = np.arange(self.func.limits.low[0], self.func.limits.high[0], step_size_x)
        plY = np.arange(self.func.limits.low[1], self.func.limits.high[1], step_size_y)
        self.X, self.Y = np.meshgrid(plX, plY, indexing='xy')
        self.Z = self.prepVals()
    def prepVals(self):
        grid_vals = [self.func(np.column_stack([self.X[itera], self.Y[itera]])) for itera in range(self.nelem)]
        return np.array(grid_vals)
    def create3d(self, showGlob = False, savePath = None):
        fig = plt.figure(figsize=(12,10))
        ax = plt.subplot(projection='3d')
        surf=ax.plot_surface(self.X, self.Y, self.Z,
                         cmap=cm.coolwarm,
                        alpha=0.7)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        [t.set_va('center') for t in ax.get_yticklabels()]
        [t.set_ha('left') for t in ax.get_yticklabels()]
        [t.set_va('center') for t in ax.get_xticklabels()]
        [t.set_ha('right') for t in ax.get_xticklabels()]
        [t.set_va('center') for t in ax.get_zticklabels()]
        [t.set_ha('left') for t in ax.get_zticklabels()]
        fig.colorbar(surf, shrink=0.35, aspect=8)
        ax.view_init(elev=15,azim=220)
        if showGlob:
            ax.scatter(self.func.glob_minma_pos[0],
                       self.func.glob_minma_pos[1],
                       self.func.glob_minma_val,
               color='black', alpha=1)
            ax.text(self.func.glob_minma_pos[0],
                       self.func.glob_minma_pos[1],
                       self.func.glob_minma_val,
                     "Global Minima",
               color='black', alpha=1)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(str(self.func))
        if savePath is not None:
            plt.savefig(savePath, dpi = 300)
        else:
            plt.show()
    def createContour(self, showGlob = False, savePath = None):
        fig = plt.figure(figsize=(12,10))
        ax = plt.subplot()
        [t.set_va('center') for t in ax.get_yticklabels()]
        [t.set_ha('left') for t in ax.get_yticklabels()]
        [t.set_va('center') for t in ax.get_xticklabels()]
        [t.set_ha('right') for t in ax.get_xticklabels()]
        ax.contourf(self.X, self.Y, self.Z)
        if showGlob:
            plt.scatter(self.func.glob_minma_pos[0], self.func.glob_minma_pos[1], color='black')
            ax.text(self.func.glob_minma_pos[0], self.func.glob_minma_pos[1], "Global Minima")
        plt.colorbar()
        if savePath is not None:
            plt.savefig(savePath, dpi = 300)
        else:
            plt.show()
```

```{code-cell} ipython3
:tags: []

plttr = Plot2dObj(StybTang2d(), 30)
```

```{code-cell} ipython3
:tags: []

plttr.Z.shape
```

```{code-cell} ipython3
:tags: []

np.testing.assert_allclose(plttr.func(np.array([plttr.X, plttr.Y])),
                    plttr.func(np.array([plttr.X.ravel(), plttr.Y.ravel()]).ravel()))
```

```{code-cell} ipython3
:tags: []

plttr.func(np.column_stack([plttr.X[0], plttr.Y[0]]).ravel())
```

```{code-cell} ipython3
:tags: []

np.column_stack([plttr.X[0], plttr.Y[0]]).ravel().reshape(-1, 2).shape
```

```{code-cell} ipython3
:tags: []

np.column_stack([plttr.X[0], plttr.Y[0]]).shape
```

```{code-cell} ipython3
:tags: []

##ff = StybTang2d()
fe = []
for itera in range(30):
    fe.append(plttr.func(np.column_stack([plttr.X[itera], plttr.Y[itera]])))
```

```{code-cell} ipython3
:tags: []

##[plttr.func(np.column_stack([plttr.X[y], plttr.Y[y]])) for y in range(30)]
```

```{code-cell} ipython3
:tags: []

# fe = []
# for x,y in zip(plttr.X, plttr.Y):
#     ##print(f"Got:\n {x}, {y}")
#     fe.append(plttr.func(np.array([0, y[0]])))
# #    for idx, sample in enumerate(x):
# #        fe.append(plttr.func(np.array([sample, y[idx]])))
#         ##print(f"Energy: {fe}")
# Z=np.array(fe).reshape(30, 30)
```

```{code-cell} ipython3
:tags: []

np.array(fe).shape
```

```{code-cell} ipython3
:tags: []

plttr.Z[-1]
```

```{code-cell} ipython3
:tags: []

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(plttr.X, plttr.Y, plttr.Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.5)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.1, aspect=5)
ax.view_init(elev=35,azim=40)

ax.scatter(-2.90353399, -2.90353399, -391.66165703062927,
           color='black', alpha=1)
#ax.text(-2.90353399, -2.90353399, -391.66165703062927, '%s' % (label), size=20, zorder=1, color='k')
plt.show()
```

```{code-cell} ipython3
:tags: []

fig = plt.figure(figsize=(12,10))
ax = plt.subplot(projection='3d')
surf=ax.plot_surface(plttr.X, plttr.Y, plttr.Z,
                     cmap=cm.coolwarm,
                    alpha=0.7)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
[t.set_va('center') for t in ax.get_yticklabels()]
[t.set_ha('left') for t in ax.get_yticklabels()]
[t.set_va('center') for t in ax.get_xticklabels()]
[t.set_ha('right') for t in ax.get_xticklabels()]
[t.set_va('center') for t in ax.get_zticklabels()]
[t.set_ha('left') for t in ax.get_zticklabels()]
fig.colorbar(surf, shrink=0.35, aspect=8)
ax.view_init(elev=15,azim=220)
ax.scatter(-2.90353399, -2.90353399, -39.16617*2,
           color='black', alpha=1)
ax.text(-2.90353399+0.4, -2.90353399, -39.16617*2, "Global Minima")
plt.xlabel('X')
plt.ylabel('Y')
plt.title("2D Styblinski-Tang")
#plt.savefig('plotp2.png', dpi = 300)
plt.show()
#print('images/plotp2.png')
```

```{code-cell} ipython3

```

```{code-cell} ipython3
:tags: []

fig = plt.figure(figsize=(12,10))
ax = plt.subplot()
[t.set_va('center') for t in ax.get_yticklabels()]
[t.set_ha('left') for t in ax.get_yticklabels()]
[t.set_va('center') for t in ax.get_xticklabels()]
[t.set_ha('right') for t in ax.get_xticklabels()]
ax.contourf(plttr.X, plttr.Y, plttr.Z)
plt.scatter(-2.90353399, -2.90353399, color='black')
ax.text(-2.90353399+0.1, -2.90353399, "Global Minima")
plt.colorbar()
plt.show()
```

```{code-cell} ipython3

```

```{code-cell} ipython3
:tags: []

plttr.create3d(showGlob=True, savePath="StybTang2d_3d.pdf")
plttr.createContour(showGlob=True, savePath="StybTang2d_contour.pdf")
```

```{code-cell} ipython3
:tags: []

ff = StybTangNd(dims=5)
```

```{code-cell} ipython3
:tags: []

ff(np.random.rand(5))
```

```{code-cell} ipython3

```
