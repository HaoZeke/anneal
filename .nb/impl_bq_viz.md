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

#!pip install pandas
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
```

```{code-cell} ipython3
:tags: []

from anneal.funcs.objNd import *
from anneal.viz.viz2d import Plot2dObj
from anneal.quenchers.boltzmann import BoltzmannQuencher
from anneal.core.components import AcceptStates, Quencher
```

```{code-cell} ipython3
:tags: []

bq = BoltzmannQuencher(StybTangNd(dims = 2), T_init=5)
```

```{code-cell} ipython3
:tags: []

bq(trackPlot=True)
```

```{code-cell} ipython3
:tags: []

bq.acceptances
```

```{code-cell} ipython3
:tags: []

bq.epoch
```

```{code-cell} ipython3
:tags: []

bq.best
```

```{code-cell} ipython3
:tags: []

bq.rejections
```

```{code-cell} ipython3
:tags: []

bq.fCalls
```

```{code-cell} ipython3
:tags: []

bq.ObjFunc.globmin
```

```{code-cell} ipython3
:tags: []

pdat=pd.DataFrame(bq.PlotData)
```

```{code-cell} ipython3
:tags: []

pdat.accept
```

```{code-cell} ipython3
:tags: []

pd.DataFrame.sample(pdat[pdat.accept==AcceptStates.REJECT]['pos'], n = 500)
```

```{code-cell} ipython3
:tags: []

len(pdat[pdat.accept==AcceptStates.REJECT]['pos'])
```

```{code-cell} ipython3
:tags: []

getDat = lambda posDat : pd.DataFrame.sample(posDat, n=nsamples) if len(posDat) > nsamples else posDat
```

```{code-cell} ipython3
:tags: []

nsamples = 500000
getDat(pdat[pdat.accept==AcceptStates.REJECT]['pos'])
```

```{code-cell} ipython3
:tags: []

# plttr = Plot2dObj(StybTangNd(2), 30)
# fig = plt.figure(figsize=(12,10))
# ax = plt.subplot()
# [t.set_va('center') for t in ax.get_yticklabels()]
# [t.set_ha('left') for t in ax.get_yticklabels()]
# [t.set_va('center') for t in ax.get_xticklabels()]
# [t.set_ha('right') for t in ax.get_xticklabels()]
# x_min = plttr.X.ravel()[plttr.Z.argmin()]
# y_min = plttr.Y.ravel()[plttr.Z.argmin()]
# ##plt.contourf(plttr.X, plttr.Y, plttr.Z, alpha=0.5)
# plt.imshow(plttr.Z, extent=[np.min(plttr.X.ravel()), np.max(plttr.X.ravel()),
#                             np.min(plttr.Y.ravel()), np.max(plttr.Y.ravel())],
#            origin='lower', cmap='viridis', alpha=0.8)
# plt.colorbar()
# ax.plot([x_min], [y_min], marker='x', markersize=5, color="white")
# ax.plot(bq.best.pos[0], bq.best.pos[1], marker='*', color='white', alpha=1)
# for point in pd.DataFrame.sample(pdat[pdat.accept==AcceptStates.REJECT]['pos'], n = 500):
#     if np.all(point > plttr.contourExtent[::2]) and np.all(point < plttr.contourExtent[1::2]):
#         p_plot = ax.scatter(point[0], point[1], marker='o', color='red', alpha=0.3)
# for point in pd.DataFrame.sample(pdat[pdat.accept==AcceptStates.IMPROVED]['pos'], n = 500):
#     if np.all(point > plttr.contourExtent[::2]) and np.all(point < plttr.contourExtent[1::2]):
#         p_plot = ax.scatter(point[0], point[1], marker='o', color='blue', alpha=0.5)
# for point in pdat[pdat.accept==AcceptStates.MHACCEPT]['pos'].to_numpy():
#     if np.all(point > plttr.contourExtent[::2]) and np.all(point < plttr.contourExtent[1::2]):
#         p_plot = ax.scatter(point[0], point[1], marker='*', color='blue', alpha=0.5)
# contours = ax.contour(plttr.X, plttr.Y, plttr.Z, 10, colors='black', alpha=0.9)
# plt.title("Boltzmann Quencher for 2D Styblinski-Tang, abs(E)<1E-3, Ti=55")
# plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
# plt.show()
# #plt.savefig("StybTang2d_contour_bquench.pdf", dpi=300)
```

```{code-cell} ipython3
:tags: []

plttr = Plot2dObj(StybTangNd(2), 30)
plttr.plotQuenchContour(bq)
```

```{code-cell} ipython3
:tags: []

plttr.contourExtent[:2]
```

```{code-cell} ipython3
:tags: []

plttr.contourExtent[::2]
```

```{code-cell} ipython3
:tags: []

plttr.contourExtent[1::2]
```

```{code-cell} ipython3

```
