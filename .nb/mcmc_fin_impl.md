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
from anneal.mcsamplers.chains import MHChain
from anneal.core.components import *
from anneal.mcsamplers.sa_chains import *
from anneal.quenchers.boltzmann import BoltzmannCooler
```

```{code-cell} ipython3
:tags: []

ff = StybTangNd(dims = 2)
mhcsa = MHChainSA(ff, BoltzmannCooler(5), n_sim = 50000)
```

```{code-cell} ipython3
:tags: []

mhcsa(np.random.default_rng().normal)
```

```{code-cell} ipython3
:tags: []

mhcsa.best
```

```{code-cell} ipython3
:tags: []

ff.globmin
```

```{code-cell} ipython3
:tags: []

ffb = lambda x: np.exp(-ff(x) / 5)
```

```{code-cell} ipython3
:tags: []

mha = MHChain(mhcsa.mk_target(5),
              np.random.default_rng().normal,
              ff.limits.mkpoint())
```

```{code-cell} ipython3
:tags: []

mha.step()
```

```{code-cell} ipython3
:tags: []

for _ in range(50000):
    mha.step()
```

```{code-cell} ipython3
:tags: []

energies = np.array([ff(x) for x in mha.states])
mha.states[energies.argmin()], min(energies)
```

```{code-cell} ipython3
:tags: []

ff.globmin
```

```{code-cell} ipython3
:tags: []

mha = MHChain(ff, Temperature=100, Nsim=10000)
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
:tags: []

mha.TargetDistrib(np.array([3,0]))
```

```{code-cell} ipython3
:tags: []

ff(np.array([3,0]))
```

```{code-cell} ipython3
:tags: []

import numpy as np

class MetropolisHastingsSampler:
    def __init__(self, target_distribution, proposal_distribution, initial_state):
        """
        Initializes the Metropolis-Hastings sampler.

        target_distribution: A function that takes a state as input and returns the unnormalized
                             probability density of the target distribution at that state.
        proposal_distribution: A function that takes a state as input and returns a new state
                               sampled from the proposal distribution.
        initial_state: The initial state of the chain.
        """
        self.target_distribution = target_distribution
        self.proposal_distribution = proposal_distribution
        self.current_state = initial_state
        self.acceptances = 0
        self.accetance_prob = 0
    def step(self):
        """
        Performs one iteration of the Metropolis-Hastings sampler.
        Returns the new state.
        """
        proposal = self.proposal_distribution(self.current_state)
        self.acceptance_prob = min(1, self.target_distribution(proposal) / self.target_distribution(self.current_state))
        if np.random.default_rng().uniform() < self.acceptance_prob:
            self.current_state = proposal
            self.acceptances += 1
        return self.current_state
```

```{code-cell} ipython3
:tags: []

import matplotlib.pyplot as plt

# Define the target distribution (a standard Gaussian)
def target_distribution(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

# Define the proposal distribution (a Gaussian with mean equal to the current state and
# standard deviation equal to 1)
def proposal_distribution(x):
    return x + np.random.default_rng().normal()

# Initialize the sampler with an initial state of 0
sampler = MetropolisHastingsSampler(target_distribution, proposal_distribution, 0)

# Run the sampler for 10000 iterations
samples = [sampler.step() for _ in range(10000)]

# Plot the histogram of the samples and the true target distribution
x = np.linspace(-5, 5, 100)
plt.hist(samples, bins=50, density=True, alpha=0.5)
plt.plot(x, target_distribution(x), color='red')
plt.show()
```

```{code-cell} ipython3
:tags: []

import matplotlib.pyplot as plt

# Define the target distribution (a standard Gaussian)
def target_distribution(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

# Define the proposal distribution (a Gaussian with mean equal to the current state and
# standard deviation equal to 1)
def proposal_distribution(x):
    return x + np.random.default_rng().normal()

# Initialize the sampler with an initial state of 0
sampler = MetropolisHastingsSampler(target_distribution, proposal_distribution, 0)

# Run the sampler for 10000 iterations and store the samples in a list
samples = []
for i in range(10000):
    sample = sampler.step()
    samples.append(sample)

# Plot the trace plot of the samples
plt.plot(samples)
plt.xlabel('Iteration')
plt.ylabel('State')
plt.show()
```

```{code-cell} ipython3
:tags: []

import matplotlib.pyplot as plt

# Define the target distribution (a standard Gaussian)
def target_distribution(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

# Define the proposal distribution (a Gaussian with mean equal to the current state and
# standard deviation equal to 1)
def proposal_distribution(x):
    return x + np.random.default_rng().normal()

# Initialize the sampler with an initial state of 0
sampler = MetropolisHastingsSampler(target_distribution, proposal_distribution, 0)

# Run the sampler for 10000 iterations and store the samples and acceptances in lists
samples = []
acceptances = []
for i in range(10000):
    sample = sampler.step()
    samples.append(sample)
    acceptances.append(np.random.default_rng().uniform() < sampler.acceptance_prob)

# Plot the trajectory with acceptances
plt.scatter(range(len(samples)), samples, c=acceptances, cmap='cool', alpha=0.5)
plt.xlabel('Iteration')
plt.ylabel('State')
plt.show()
```

```{code-cell} ipython3
:tags: []

import numpy as np

class SimulatedAnnealing:
    def __init__(self, target_distribution, proposal_distribution, initial_state, temperature_schedule):
        self.target_distribution = target_distribution
        self.proposal_distribution = proposal_distribution
        self.state = initial_state
        self.temperature_schedule = temperature_schedule

    def step(self):
        # Get the current temperature from the temperature schedule
        current_temperature = self.temperature_schedule.get_current_temperature()

        # Generate a proposal from the proposal distribution
        proposed_state = self.proposal_distribution(self.state)

        # Calculate the acceptance probability
        acceptance_prob = min(1, np.exp((self.target_distribution(proposed_state) -
                                          self.target_distribution(self.state)) / current_temperature))

        # Accept or reject the proposal based on the acceptance probability
        if np.random.default_rng().uniform() < acceptance_prob:
            self.state = proposed_state
            accepted = True
        else:
            accepted = False

        # Update the temperature schedule
        self.temperature_schedule.update()

        # Return the current state and whether the proposal was accepted
        return self.state, accepted
```

```{code-cell} ipython3
:tags: []

class LinearTemperatureSchedule:
    def __init__(self, initial_temperature, final_temperature, cooling_schedule):
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_schedule = cooling_schedule
        self.current_iteration = 0

    def get_current_temperature(self):
        return self.initial_temperature + (self.final_temperature - self.initial_temperature) * self.current_iteration / self.cooling_schedule

    def update(self):
        self.current_iteration += 1
```

```{code-cell} ipython3
:tags: []

# Define the target distribution (a standard Gaussian)
def target_distribution(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

# Define the proposal distribution (a Gaussian with mean equal to the current state and
# standard deviation equal to 1)
def proposal_distribution(x):
    return x + np.random.default_rng().normal()

# Initialize the simulated annealing algorithm with an initial state of 0 and a linear temperature schedule
sa = SimulatedAnnealing(target_distribution, proposal_distribution, 0, LinearTemperatureSchedule(10, 0.01, 10000))

# Run the algorithm for 10000 iterations and store
```

```{code-cell} ipython3
:tags: []

# Run the algorithm for 10000 iterations and store the resulting states
n_iterations = 10000
states = np.zeros(n_iterations)
acceptance = np.zeros(n_iterations)
for i in range(n_iterations):
    states[i], acceptance[i] = sa.step()
```

```{code-cell} ipython3
:tags: []

import matplotlib.pyplot as plt

# Plot the trajectory and acceptance rate
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

ax1.plot(states)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('State')
ax1.set_title('Trajectory of Simulated Annealing Sampler')

ax2.plot(acceptance)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Acceptance Rate')
ax2.set_ylim([-0.1, 1.1])
ax2.set_title('Acceptance Rate of Simulated Annealing Sampler')

plt.tight_layout()
plt.show()
```

```{code-cell} ipython3

```
