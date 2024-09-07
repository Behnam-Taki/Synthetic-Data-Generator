from typing import Any, Self, Dict, List

from .base import Distribution
from .general import MarkovDistribution, PoissonDistribution, CategoricalDistribution


class WordDistribution(Distribution):
    def __init__(self, character_markov_dist: MarkovDistribution, length_dist: PoissonDistribution):
        self.character_markov_dist = character_markov_dist
        self.length_dist = length_dist

    @classmethod
    def get_random_distribution(cls, *args, **kwargs) -> Self:
        states = [chr(i + ord('a')) for i in range(26)] + ['$']
        return WordDistribution(character_markov_dist=MarkovDistribution.get_random_distribution(states=states),
                                length_dist=PoissonDistribution.get_random_distribution())

    def generate_sample(self, *args, **kwargs) -> Any:
        char_sequence = self.character_markov_dist.generate_sample(lenght=self.length_dist.generate_sample() + 1)
        return ''.join(char_sequence)

    def generate_child(self, distance: float) -> Self:
        pass


class WordPosDistribution(Distribution):
    def __init__(self,
                 word_distribution: WordDistribution,
                 pos_distribution: CategoricalDistribution,
                 wordset_size: int):
        self.word_distribution = word_distribution
        self.pos_distribution = pos_distribution
        self.wordset_size = wordset_size
        pos_wordsets: Dict[str, List] = {pos: [] for pos in pos_distribution.states}
        empty_pos = set(pos_wordsets.keys())
        for i in range(wordset_size):
            if i + len(empty_pos) >= wordset_size:
                break
            word = self.word_distribution.generate_sample()
            pos = self.pos_distribution.generate_sample()
            pos_wordsets[pos].append(word)
            if pos in empty_pos:
                empty_pos.remove(pos)
        for pos in empty_pos:
            word = self.word_distribution.generate_sample()
            pos_wordsets[pos].append(word)
        self.pos_word_dists: Dict[str, CategoricalDistribution] = {
            pos: CategoricalDistribution.get_random_distribution(states=word_list)
            for pos, word_list in pos_wordsets.items()
        }

    @classmethod
    def get_random_distribution(cls, pos_list: List[str], wordset_size=None) -> Self:
        wordset_size = wordset_size or PoissonDistribution(500).generate_sample()
        return WordPosDistribution(
            word_distribution=WordDistribution.get_random_distribution(),
            pos_distribution=CategoricalDistribution.get_random_distribution(states=pos_list),
            wordset_size=wordset_size
        )

    def generate_sample(self, pos: str) -> Any:
        return self.pos_word_dists[pos].generate_sample()

    def generate_child(self, distance: float) -> Self:
        pass
