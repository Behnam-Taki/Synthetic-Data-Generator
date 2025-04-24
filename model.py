from __future__ import annotations
import pickle
from typing import List, Self

from distribution.document import WordPosDistribution, DocumentDistribution
from utilities import to_sub_description


class SyntheticGeneratorModel:
    default_file_name = 'synthetic_generator.model'

    def __init__(self, pos_list: List[str], pos_word_dist_count: int):
        self.pos_list = pos_list
        self.pos_word_dist_count = pos_word_dist_count
        self.wordpos_dists = [
            WordPosDistribution.get_random_distribution(pos_list=self.pos_list, wordset_size=500)
            for _ in range(self.pos_word_dist_count)
        ]
        self.root_document_distribution = DocumentDistribution.get_random_distribution(wordpos_dists=self.wordpos_dists)

    def describe(self) -> str:
        wordpos_dists = '\n'.join([
            f'{type(self.wordpos_dists[i]).__name__} #{i + 1}: {to_sub_description(self.wordpos_dists[i].describe())}'
            for i in range(len(self.wordpos_dists))])
        return f'Word-Pos Distributions:{to_sub_description(wordpos_dists)}\n' \
               f'Root Document Distribution:{to_sub_description(self.root_document_distribution.describe())}'

    def save(self, path: str = default_file_name):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str = default_file_name) -> SyntheticGeneratorModel:
        with open(path, 'rb') as f:
            return pickle.load(f)

