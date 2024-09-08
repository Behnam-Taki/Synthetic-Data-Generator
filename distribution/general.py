import math
from typing import List, Self, TypeVar, Generic
from tabulate import tabulate

import numpy as np

from .base import Distribution


class GaussianDistribution(Distribution):
    def __init__(self, mean: float, variance: float):
        self.mean = mean
        self.variance = variance

    @classmethod
    def get_random_distribution(cls, **kwargs) -> Self:
        return GaussianDistribution(
            mean=np.random.normal(loc=0, scale=1),
            variance=np.random.normal(loc=2, scale=1)
        )

    def generate_sample(self, **kwargs) -> float:
        return np.random.normal(loc=self.mean, scale=math.sqrt(self.variance))

    def generate_child(self, distance: float) -> Self:
        raise NotImplementedError()

    def describe(self) -> str:
        return f'Gaussian({self.mean:.3f}, {self.variance:.3f})'


class PoissonDistribution(Distribution):
    def __init__(self, lam: float):
        self.lam = lam

    @classmethod
    def get_random_distribution(cls, **kwargs) -> Self:
        return PoissonDistribution(lam=np.random.gamma(shape=10, scale=0.5))

    def generate_sample(self, **kwargs) -> int:
        return np.random.poisson(lam=self.lam)

    def generate_child(self, distance: float) -> Self:
        raise NotImplementedError()

    def describe(self) -> str:
        return f'Poisson({self.lam:.3f})'


CategoryType = TypeVar('CategoryType')


class CategoricalDistribution(Distribution, Generic[CategoryType]):
    def __init__(self, states: List[CategoryType], probabilities: np.ndarray):
        self.states = states
        self.probabilities = probabilities
        self._cdf = self.probabilities.cumsum()

    @classmethod
    def get_random_distribution(cls, states: List[CategoryType], **kwargs) -> Self:
        return CategoricalDistribution(states, np.random.dirichlet(np.ones(len(states))))

    def generate_sample(self, **kwargs) -> CategoryType:
        r = np.random.uniform()
        index = np.searchsorted(self._cdf, r)
        return self.states[index]

    def generate_child(self, distance: float) -> Self:
        raise NotImplementedError()

    def describe(self) -> str:
        col_count = 10
        states_tuples = [(self.states[i], self.probabilities[i]) for i in range(len(self.states))] \
            if type(self.states[0]).__str__ is not object.__str__ else \
            [(f'{type(self.states[0]).__name__} #{i}', self.probabilities[i]) for i in range(len(self.states))]
        states_tuples.sort(key=lambda x: len(x[0]))
        state_probs = [f'{state}: {prob:.3%}' for state, prob in states_tuples]
        line_count = math.ceil(len(state_probs) / col_count)
        state_probs_str = tabulate([[
            state_probs[l + line_count * c] for c in range(col_count) if l + line_count * c < len(state_probs)]
            for l in range(line_count)
        ], tablefmt="plain")
        return f'Categorical{self._to_sub_description(state_probs_str, format_for_one_line=True)}'


StateType = TypeVar('StateType')


class MarkovDistribution(Distribution, Generic[StateType]):
    def __init__(self, states: List[StateType],
                 initial_distribution: CategoricalDistribution[StateType],
                 transition_distributions: List[CategoricalDistribution[StateType]],
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
    def get_random_distribution(cls, states: List[StateType], final_state_index=None, **kwargs) -> Self:
        indices_list = list(range(len(states)))
        initialable_indices_list = indices_list.copy()
        initialable_indices_list.pop(final_state_index if final_state_index is not None else -1)
        return MarkovDistribution(
            states,
            CategoricalDistribution.get_random_distribution(initialable_indices_list),
            [CategoricalDistribution.get_random_distribution(indices_list) for _ in states]
        )

    def _generate_next_state(self, current_state_index: int = None, allow_final=True) -> int:
        next_state_distribution: CategoricalDistribution[StateType] = (
            self.transition_distributions[current_state_index] if current_state_index else
            self.initial_distribution)
        while True:
            result = next_state_distribution.generate_sample()
            if allow_final or result != self.final_state_index:
                return result

    def generate_sample(self, lenght: int = None, **kwargs) -> List[StateType]:
        sample = []
        allow_final = lenght is None
        current_state_index = self._generate_next_state(allow_final=allow_final)
        while current_state_index != self.final_state_index and (allow_final or len(sample) < lenght):
            sample.append(self.states[current_state_index])
            current_state_index = self._generate_next_state(current_state_index, allow_final=allow_final)
        return sample

    def generate_child(self, distance: float) -> Self:
        raise NotImplementedError()

    def describe(self) -> str:
        trans_mat = tabulate([['-->'] + [100 * p for p in self.initial_distribution.probabilities]] +
                             [[self.states[i]] + [100 * p for p in self.transition_distributions[i].probabilities]
                              for i in range(len(self.states))],
                             headers=[''] + self.states, floatfmt='4.1f')
        return f'Markov{self._to_sub_description(trans_mat)}'
