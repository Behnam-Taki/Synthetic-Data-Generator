from __future__ import annotations
import pickle
from queue import Queue
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

    def derivate(self, max_derivation_level: int, branching_factor: int = 2,
                 initial_distance_scale: float = 2, decay_factor: float = 3):
        current_queue = Queue()
        current_queue.put(self.root_document_distribution)
        current_distance_scale = initial_distance_scale
        for level in range(max_derivation_level):
            new_queue = Queue()
            while not current_queue.empty():
                dist: DocumentDistribution = current_queue.get()
                for i in range(branching_factor):
                    derivated = dist.derivate(current_distance_scale)
                    new_queue.put(derivated)
            current_queue = new_queue
            current_distance_scale /= decay_factor

    def describe(self) -> str:
        wordpos_dists = '\n'.join([
            f'{type(self.wordpos_dists[i]).__name__} #{i + 1}: {to_sub_description(self.wordpos_dists[i].describe())}'
            for i in range(len(self.wordpos_dists))])
        return f'Word-Pos Distributions:{to_sub_description(wordpos_dists)}\n' \
               f'{"━" * 80}\n' \
               f'{self.root_document_distribution.describe()}'

    def save(self, path: str = default_file_name):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str = default_file_name) -> SyntheticGeneratorModel:
        with open(path, 'rb') as f:
            return pickle.load(f)

