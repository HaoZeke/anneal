import numpy as np


class MHChain:
    """
    Implements a Markov Chain using the Metropolis-Hastings algorithm.

    #### Notes
    The Metropolis-Hastings algorithm is a method used to sample from a
    probability distribution by constructing a Markov chain.  It provides a
    procedure to move from the current state to a new state in the Markov chain.

    In this class, the Metropolis-Hastings algorithm is used to construct a
    Markov chain by sampling from a target distribution.
    """

    def __init__(self, Target, Proposal, InitialState):
        """
        Initializes an instance of the `MHChain` class.

        #### Parameters
        **Target**
        : The target distribution from which to sample. This is an un-normalized
        distribution function.

        **Proposal**
        : The proposal distribution used to suggest new states in the Markov
        chain.

        **InitialState**
        : The initial state of the Markov chain.
        """
        self.target = Target
        self.proposal = Proposal
        self.cstate = InitialState
        self.states = []

    def step(self):
        """
        Performs one step in the Markov chain using the Metropolis-Hastings
        algorithm.

        In each step, a proposed state is drawn from the proposal distribution.
        This state is accepted as the next state in the chain with a probability
        given by the ratio of the target distribution at the proposed state and
        the current state.  If the proposed state is not accepted, the current
        state is repeated in the chain.
        """
        prop = self.proposal(self.cstate)
        aproba = min(1, self.target(prop) / self.target(self.cstate))
        if np.random.default_rng().uniform() < aproba:
            self.cstate = prop
        self.states.append(self.cstate)
