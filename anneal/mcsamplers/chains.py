from anneal.core.components import *


class MHChain:
    def __init__(self, Target, Proposal, InitialState):
        """
        Metropolis Hastings Chain sampler
        Target: Un-normalized
        """
        self.target = Target
        self.proposal = Proposal
        self.cstate = InitialState
        self.states = []

    def step(self):
        prop = self.proposal(self.cstate)
        aproba = min(1, self.target(prop) / self.target(self.cstate))
        if np.random.default_rng().uniform() < aproba:
            self.cstate = prop
        self.states.append(self.cstate)
