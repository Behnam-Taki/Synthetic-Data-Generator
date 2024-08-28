from typing import List, Any, Self

import numpy as np

from .base import Distribution


class CategoricalDistribution(Distribution):
    def __init__(self, states: List[Any], probabilities: np.ndarray):
        self.states = states
        self.probabilities = probabilities
        self._cdf = self.probabilities.cumsum()

    @classmethod
    def get_random_distribution(cls, states: List[Any]) -> Self:
        return CategoricalDistribution(states, np.random.dirichlet(np.ones(len(states)), size=1))

    def generate_sample(self) -> Any:
        r = np.random.uniform()
        index = np.searchsorted(self._cdf, r)
        return self.states[index]

    def generate_child(self, distance: float) -> Self:
        raise NotImplementedError()


class MarkovDistribution(Distribution):
    def __init__(self, states: List[Any],
                 initial_distribution: CategoricalDistribution,
                 transition_distributions: List[CategoricalDistribution],
                 final_state_index=None
                 ):
        """
        :param states: A list of all states including the final state
        :param initial_distribution: The categorical distribution on state indices
        :param transition_distributions: A list of categorical distribution on state indices, each for transisions from
            one of the states
        :param final_state_index: The index of the final state, default is the last state
        """
        self.states = states
        self.initial_distribution = initial_distribution
        self.transition_distributions = transition_distributions
        self.final_state_index = final_state_index if final_state_index is not None else len(states) - 1

    @classmethod
    def get_random_distribution(cls, states: List[Any], final_state_index=None) -> Self:
        indices_list = list(range(len(states)))
        initialable_indices_list = indices_list.copy()
        initialable_indices_list.pop(final_state_index if final_state_index is not None else -1)
        return MarkovDistribution(
            states,
            CategoricalDistribution.get_random_distribution(initialable_indices_list),
            [CategoricalDistribution.get_random_distribution(indices_list) for _ in states]
        )

    def generate_sample(self) -> List[Any]:
        sample = []
        current_state_index = self.initial_distribution.generate_sample()
        while current_state_index != self.final_state_index:
            sample.append(self.states[current_state_index])
            current_state_index = self.transition_distributions[current_state_index].generate_sample()
        return sample

    def generate_child(self, distance: float) -> Self:
        pass
