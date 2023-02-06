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

import numpy as np
import pandas as pd
import scipy.stats as st

from anneal.funcs.obj2d import *
from anneal.funcs.objNd import *
```

```{code-cell} ipython3
:tags: []

from anneal.core.components import *

from anneal.core.components import *


class BaseChain(metaclass=abc.ABCMeta):
    """The Base chain for MCMC samplers used in SA"""

    def __init__(
        self,
        ObjFunc: ObjectiveFunction,
        Temperature,
        Nsim=5000,
    ):
        """A key point is that the temperature is used to define the target probability distribution"""
        self.ObjFunc = ObjFunc
        self.Nsim = Nsim
        self.Temperature = Temperature
        self.stepNum = 0
        self.states = np.array([self.ObjFunc.limits.mkpoint() for x in range(self.Nsim)])
        self.status = np.empty(self.Nsim, dtype=bool)
        self.fvals = np.zeros(self.Nsim)
        ## This is the probability at each temperature
        self.TargetDistrib = lambda point: np.exp(
            -self.ObjFunc(point) / self.Temperature
        )


class MHChain(BaseChain):
    def __init__(self, ObjFunc: ObjectiveFunction, Temperature, Nsim=5000):
        super().__init__(ObjFunc, Temperature, Nsim)
        self.states[0] = np.zeros(ObjFunc.limits.dims)
        self.status[0] = True
        self.cand = None
        self.cur_state = self.states[0]

    def step(self):
        """Here we pick a candidate which is chain-independent"""
        for step in range(1, self.Nsim):
            cand = self.ObjFunc.limits.mkpoint()
            rho = self.bounds_eval_rho(cand, step - 1)
            if self.proposal() < rho:
                self.states[step] = self.states[step - 1] + (
                    cand - self.states[step]
                )
                self.status[step] = True
            else:
                self.states[step] = self.states[step - 1]
                self.status[step] = False

    def bounds_eval_rho(self, cand, chain_id):
        try:
            val = self.TargetDistrib(self.states[chain_id])
        except OutOfBounds:
            self.states[chain_id] = self.ObjFunc.limits.clip(
                self.states[chain_id]
            )
            val = self.TargetDistrib(self.states[chain_id])
        return self.TargetDistrib(cand) / val

    def proposal(self):
        return np.random.uniform(0, 1, 1)
```

```{code-cell} ipython3
:tags: []

ff = StybTangNd(dims = 20)
```

```{code-cell} ipython3
:tags: []

mha = MHChain(ff, Temperature=20, Nsim=9000)
```

```{code-cell} ipython3
:tags: []

mha.step()
```

```{code-cell} ipython3
:tags: []

np.sum(mha.status) / len(mha.status)
```

```{code-cell} ipython3
:tags: []

energies = np.array([ff(x) for x in mha.states[:-1]])
min_state = mha.states[energies.argmin()]
FPair(pos = min_state, val = np.min(energies))
```

```{code-cell} ipython3
:tags: []

ff.globmin
```

```{code-cell} ipython3

```
