import math
import random
from typing import List, Self, TypeVar, Generic, Any, Callable
from tabulate import tabulate

import numpy as np

from utilities import to_sub_description, bin_search_on_answer
from .base import Distribution


def _reject_nonpositive(func: Callable, value_to_check: Callable = lambda x: x, *args, **kwargs):
    while True:
        result = func(*args, **kwargs)
        if value_to_check(result) > 0:
            return result


class GaussianDistribution(Distribution):
    def __init__(self, mean: float, variance: float):
        self.mean = mean
        self.variance = variance

    @classmethod
    def get_random_distribution(cls, **kwargs) -> Self:
        return GaussianDistribution(
            mean=np.random.normal(loc=0, scale=1),
            variance=_reject_nonpositive(np.random.normal, loc=2, scale=1)
        )

    def generate_sample(self, **kwargs) -> float:
        return np.random.normal(loc=self.mean, scale=math.sqrt(self.variance))

    def derivate(self, distance_scale: float) -> Self:
        kl = 0.5 * distance_scale
        if kl == 0:
            return GaussianDistribution(self.mean, self.variance)
        distance_with_same_mean_calculator = lambda var_ratio: 0.5 * math.log(var_ratio) + 0.5 / var_ratio - 0.5

        max_var_ratio_range_end = 1
        while distance_with_same_mean_calculator(max_var_ratio_range_end) < kl:
            max_var_ratio_range_end *= 2
        max_var_ratio_range_start = max_var_ratio_range_end / 2
        max_var_ratio = bin_search_on_answer(distance_with_same_mean_calculator, kl,
                                             max_var_ratio_range_start, max_var_ratio_range_end)
        max_var = max_var_ratio * self.variance

        # min_var_ratio_range_start = 1
        # while distance_with_same_mean_calculator(min_var_ratio_range_start) < distance:
        #     min_var_ratio_range_start /= 2
        # min_var_ratio_range_end = min_var_ratio_range_start / 2
        # min_var_ratio = bin_search_on_answer(distance_with_same_mean_calculator, distance, min_var_ratio_range_start, min_var_ratio_range_end) + 1e-3
        # min_var = min_var_ratio * self.variance

        min_var = 0
        mu_calculator = lambda var: self.mean + math.sqrt(
            2 * var * (kl + 0.5 - 0.5 * math.log(var / self.variance) - 0.5 * self.variance / var))
        parts_num = 1000
        epsilon = (max_var - min_var) / 1000
        part_lenghts: np.ndarray = np.zeros(parts_num)
        for part in range(parts_num):
            try:
                part_lenght = math.sqrt(epsilon ** 2 + (mu_calculator(min_var + part * epsilon) - mu_calculator(
                    min_var + (part + 1) * epsilon)) ** 2)
            except:
                part_lenght = 0
            part_lenghts[part] = part_lenght
        var_dist = CategoricalDistribution(list(range(parts_num)), part_lenghts / sum(part_lenghts))
        selected_var_part = var_dist.generate_sample()
        selected_var = random.uniform(min_var + selected_var_part * epsilon,
                                      min_var + (selected_var_part + 1) * epsilon)
        positive_selected_mu = mu_calculator(selected_var)
        selected_mu = random.choice([positive_selected_mu, 2 * self.mean - positive_selected_mu])
        return GaussianDistribution(selected_mu, selected_var)

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

    def derivate(self, distance_scale: float) -> Self:
        kl = 1 * distance_scale
        distance_calculator = lambda lam: lam - self.lam + self.lam * math.log(self.lam / lam)
        first_range_start, sec_range_end = self.lam, self.lam
        while distance_calculator(first_range_start) < kl:
            first_range_start /= 2
        while distance_calculator(sec_range_end) < kl:
            sec_range_end *= 2
        start = bin_search_on_answer(distance_calculator, kl, first_range_start, self.lam)
        end = bin_search_on_answer(distance_calculator, kl, self.lam, sec_range_end)
        lam = random.choice([start, end])
        return PoissonDistribution(lam)

    def describe(self) -> str:
        return f'Poisson({self.lam:.3f})'


CategoryType = TypeVar('CategoryType')


class CategoricalDistribution(Distribution, Generic[CategoryType]):
    def __init__(self, states: List[CategoryType], probabilities: np.ndarray):
        self.states = states
        self.probabilities = probabilities
        self._cdf = self.probabilities.cumsum()

    @classmethod
    def get_random_distribution(cls, states: List[CategoryType], prior_dirichlet_params=None, **kwargs) -> Self:
        prior_dirichlet_params = prior_dirichlet_params if prior_dirichlet_params is not None else np.ones(len(states))
        return CategoricalDistribution(states, np.random.dirichlet(prior_dirichlet_params))

    def generate_sample(self, **kwargs) -> CategoryType:
        r = np.random.uniform()
        index = np.searchsorted(self._cdf, r)
        return self.states[index]

    def derivate(self, distance_scale: float) -> Self:
        if distance_scale == 0:
            return CategoricalDistribution(states=self.states.copy(), probabilities=self.probabilities.copy())
        kl = distance_scale * 0.3
        n = len(self.states)
        ps = self.probabilities
        entropy = - sum(pi * math.log(pi) for pi in ps)
        c = kl + entropy
        delta = 0.9
        epsilon = 1e-6
        min_p = 0.01
        while True:
            non_normalized_qs = [random.uniform(0, 1) for _ in range(n-1)]
            normalization_factor = (1 - ps[-1]) / sum(non_normalized_qs)
            qs = [q * normalization_factor for q in non_normalized_qs]
            ts = [- ps[i] * math.log(qs[i]) for i in range(len(qs))]
            tn = c - sum(ts)
            if tn < 0:
                continue
            qn = math.exp(- tn / ps[-1])
            if qn < min_p:
                continue
            if abs(sum(qs) + qn - 1) > delta:
                continue
            # noinspection PyTypeChecker
            function_index = int(np.argmax(qs + [qn]))
            if function_index < n - 1:
                qn, qs[function_index] = qs[function_index], qn
                ps[-1], ps[function_index] = ps[function_index], ps[-1]
            while True:
                qn = math.exp(-(c + sum(ps[i] * math.log(qs[i]) for i in range(len(qs)))) / ps[-1])
                f = sum(qs) + qn - 1
                if -epsilon <= f <= epsilon:
                    if function_index < n-1:
                        qn, qs[function_index] = qs[function_index], qn
                        ps[-1], ps[function_index] = ps[function_index], ps[-1]
                    new_qs = np.array(qs + [qn])
                    new_qs = new_qs / np.sum(new_qs)
                    return CategoricalDistribution(states=self.states.copy(), probabilities=new_qs)
                grads = [1 - (qn * ps[j]) / (ps[-1] * qs[j]) for j in range(len(qs))]
                for i in range(len(qs)):
                    if qs[i] <= min_p and f * grads[i] > 0:
                        grads[i] = 0
                grads_norm = np.sum(np.square(grads))
                if grads_norm == 0:
                    break
                qs = [qs[i] - f * grads[i] / grads_norm for i in range(len(qs))]
                if any(x <= 0 for x in qs):
                    break

    def describe(self) -> str:
        col_count = 10
        states_tuples = [(self.states[i], self.probabilities[i]) for i in range(len(self.states))] \
            if type(self.states[0]).__str__ is not object.__str__ or \
               type(self.states[0]) in {GaussianDistribution, PoissonDistribution} else \
            [(f'{type(self.states[0]).__name__} #{i + 1}', self.probabilities[i]) for i in range(len(self.states))]
        states_tuples.sort(key=lambda x: len(str(x[0])))
        state_probs = [f'{state}: {prob:.3%}' for state, prob in states_tuples]
        line_count = math.ceil(len(state_probs) / col_count)
        state_probs_str = tabulate([[
            state_probs[l + line_count * c] for c in range(col_count) if l + line_count * c < len(state_probs)]
            for l in range(line_count)
        ], tablefmt="plain")
        return f'Categorical{to_sub_description(state_probs_str, format_for_one_line=True)}'


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

    def derivate(self, distance_scale: float) -> Self:
        return MarkovDistribution(
            states=self.states,
            initial_distribution=self.initial_distribution.derivate(distance_scale=distance_scale),
            transition_distributions=[dist.derivate(distance_scale=distance_scale) for dist in
                                      self.transition_distributions],
            final_state_index=self.final_state_index
        )

    def describe(self) -> str:
        trans_mat = tabulate([['-->'] + [100 * p for p in self.initial_distribution.probabilities]] +
                             [[self.states[i]] + [100 * p for p in self.transition_distributions[i].probabilities]
                              for i in range(len(self.states))],
                             headers=[''] + self.states, floatfmt='4.1f')
        return f'Markov{to_sub_description(trans_mat)}'


class PositiveGaussianMixtureDistribution(Distribution):
    def __init__(self, gaussian_dist_distribution: CategoricalDistribution[GaussianDistribution]):
        self.gaussian_dist_distribution = gaussian_dist_distribution

    @classmethod
    def get_random_distribution(cls,
                                gaussians_count: int = 2,
                                smallest_scale: int = 1,
                                scale_factor: float = 2,
                                prior_dirichlet_params: List[float] = None,
                                **kwargs) -> Self:
        prior_dirichlet_params = prior_dirichlet_params if prior_dirichlet_params else [1] * gaussians_count
        gaussians = []
        for i in range(gaussians_count):
            random_gaussian = GaussianDistribution.get_random_distribution()
            random_gaussian.mean += smallest_scale * scale_factor ** i
            gaussians.append(random_gaussian)
        return PositiveGaussianMixtureDistribution(CategoricalDistribution.get_random_distribution(
            states=gaussians,
            prior_dirichlet_params=np.array(prior_dirichlet_params)
        ))

    def generate_sample(self, **kwargs) -> Any:
        return _reject_nonpositive(self.gaussian_dist_distribution.generate_sample().generate_sample)

    def derivate(self, distance_scale: float) -> Self:
        new_gaussian_dists = \
            [_reject_nonpositive(dist.derivate, lambda d: d.mean, distance_scale= distance_scale)
             for dist in self.gaussian_dist_distribution.states]
        new_categorical_dist = self.gaussian_dist_distribution.derivate(distance_scale=distance_scale)
        new_categorical_dist.states = new_gaussian_dists
        return PositiveGaussianMixtureDistribution(gaussian_dist_distribution=new_categorical_dist)

    def describe(self) -> str:
        return self.gaussian_dist_distribution.describe()
