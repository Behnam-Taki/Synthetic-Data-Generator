from typing import List, Self

from distribution.text import WordPosDistribution, TextDistribution
from utilities import to_sub_description


class SyntheticGeneratorModel:
    def __init__(self, pos_list: List[str], pos_word_dist_count: int):
        self.pos_list = pos_list
        self.pos_word_dist_count = pos_word_dist_count
        self.wordpos_dists = [
            WordPosDistribution.get_random_distribution(pos_list=self.pos_list, wordset_size=500)
            for _ in range(self.pos_word_dist_count)
        ]
        self.root_text_distribution = TextDistribution.get_random_distribution(wordpos_dists=self.wordpos_dists)

    def describe(self) -> str:
        wordpos_dists = '\n'.join([
            f'{type(self.wordpos_dists[i]).__name__} #{i + 1}: {to_sub_description(self.wordpos_dists[i].describe())}'
            for i in range(len(self.wordpos_dists))])
        return f'Word-Pos Distributions:{to_sub_description(wordpos_dists)}\n' \
               f'Root Text Distribution:{to_sub_description(self.root_text_distribution.describe())}'

    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str) -> Self:
        pass
