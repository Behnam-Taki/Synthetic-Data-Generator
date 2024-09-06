import math
from typing import List, Any, Self

import numpy as np

from .base import Distribution


class GaussianDistribution(Distribution):
    def __init__(self, mean: float, variance: float):
        self.mean = mean
        self.variance = variance

    @classmethod
    def get_random_distribution(cls) -> Self:
        return GaussianDistribution(
            mean=np.random.normal(loc=0, scale=1),
            variance=np.random.normal(loc=2, scale=1)
        )

    def generate_sample(self, *args, **kwargs) -> Any:
        return np.random.normal(loc=self.mean, scale=math.sqrt(self.variance))

    def generate_child(self, distance: float) -> Self:
        raise NotImplementedError()


class PoissonDistribution(Distribution):
    def __init__(self, lam: float):
        self.lam = lam

    @classmethod
    def get_random_distribution(cls) -> Self:
        return PoissonDistribution(lam=np.random.gamma(shape=10, scale=0.5))

    def generate_sample(self, *args, **kwargs) -> Any:
        return np.random.poisson(lam=self.lam)

    def generate_child(self, distance: float) -> Self:
        raise NotImplementedError()


class CategoricalDistribution(Distribution):
    def __init__(self, states: List[Any], probabilities: np.ndarray):
        self.states = states
        self.probabilities = probabilities
        self._cdf = self.probabilities.cumsum()

    @classmethod
    def get_random_distribution(cls, states: List[Any]) -> Self:
        return CategoricalDistribution(states, np.random.dirichlet(np.ones(len(states)), size=1))

    def generate_sample(self, *args, **kwargs) -> Any:
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

    def _generate_next_state(self, current_state_index: int = None, allow_final=True):
        next_state_distribution = (
            self.transition_distributions[current_state_index] if current_state_index else
            self.initial_distribution)
        while True:
            result = next_state_distribution.generate_sample()
            if allow_final or result != self.final_state_index:
                return result

    def generate_sample(self, lenght: int = None) -> List[Any]:
        sample = []
        allow_final = lenght is None
        current_state_index = self._generate_next_state(allow_final=allow_final)
        while current_state_index != self.final_state_index and (allow_final or len(sample) < lenght):
            sample.append(self.states[current_state_index])
            current_state_index = self._generate_next_state(current_state_index, allow_final=allow_final)
        return sample

    def generate_child(self, distance: float) -> Self:
        pass
