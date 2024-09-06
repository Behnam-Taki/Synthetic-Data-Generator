from typing import Any, Self

from .base import Distribution
from .general import MarkovDistribution, PoissonDistribution


class WordDistribution(Distribution):
    def __init__(self, character_markov_dist: MarkovDistribution, length_dist: PoissonDistribution):
        self.character_markov_dist = character_markov_dist
        self.length_dist = length_dist

    @classmethod
    def get_random_distribution(cls, *args, **kwargs) -> Self:
        states = [chr(i+ord('a')) for i in range(26)] + ['$']
        return WordDistribution(character_markov_dist=MarkovDistribution.get_random_distribution(states=states),
                                length_dist=PoissonDistribution.get_random_distribution())

    def generate_sample(self, *args, **kwargs) -> Any:
        char_sequence = self.character_markov_dist.generate_sample(lenght=self.length_dist.generate_sample() + 1)
        return ''.join(char_sequence)

    def generate_child(self, distance: float) -> Self:
        pass
