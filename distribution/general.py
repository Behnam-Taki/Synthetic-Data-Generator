from typing import List, Any

import numpy as np

from .base import Distribution


class CategoricalDistribution(Distribution):
    def __init__(self, states: List[Any], probabilities: np.ndarray):
        self.states = states
        self.probabilities = probabilities
        self._cdf = self.probabilities.cumsum()

    @classmethod
    def get_random_distribution(cls, states: List[Any]) -> Distribution:
        return CategoricalDistribution(states, np.random.dirichlet(np.ones(len(states)), size=1))

    def generate_sample(self) -> Any:
        r = np.random.uniform()
        index = np.searchsorted(self._cdf, r)
        return self.states[index]

    def generate_child(self, distance: float) -> Distribution:
        pass
