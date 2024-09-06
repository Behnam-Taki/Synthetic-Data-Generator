from typing import Any, Self

from .base import Distribution
from .general import MarkovDistribution


class WordDistribution(Distribution):
    def __init__(self, character_markov_dist: MarkovDistribution):
        self.character_markov_dist = character_markov_dist

    @classmethod
    def get_random_distribution(cls, *args, **kwargs) -> Self:
        states = [chr(i+ord('a')) for i in range(26)] + ['$']
        return WordDistribution(character_markov_dist=MarkovDistribution.get_random_distribution(states=states))

    def generate_sample(self, *args, **kwargs) -> Any:
        char_sequence = self.character_markov_dist.generate_sample()
        return ''.join(char_sequence)

    def generate_child(self, distance: float) -> Self:
        pass
