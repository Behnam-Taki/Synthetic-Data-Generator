from abc import ABC, abstractmethod
from typing import Self, Dict, List, TypeVar, Generic, Set, Type, Tuple

from utilities import to_sub_description
from .base import Distribution
from .general import MarkovDistribution, PoissonDistribution, CategoricalDistribution


class WordDistribution(Distribution):
    def __init__(self,
                 character_markov_dist: MarkovDistribution[str],
                 length_dist: PoissonDistribution):
        self.character_markov_dist = character_markov_dist
        self.length_dist = length_dist

    @classmethod
    def get_random_distribution(cls, **kwargs) -> Self:
        states = [chr(i + ord('a')) for i in range(26)] + ['$']
        return WordDistribution(character_markov_dist=MarkovDistribution.get_random_distribution(states=states),
                                length_dist=PoissonDistribution.get_random_distribution())

    def generate_sample(self, **kwargs) -> str:
        char_sequence = self.character_markov_dist.generate_sample(lenght=self.length_dist.generate_sample() + 1)
        return ''.join(char_sequence)

    def derivate(self, distance_scale: float) -> Self:
        raise NotImplementedError()

    def describe(self) -> str:
        return f'Charachter Count Distribution:{to_sub_description(self.length_dist.describe())}\n' \
               f'Character Distribution:{to_sub_description(self.character_markov_dist.describe())}'


PosType = TypeVar('PosType')


class WordPosDistribution(Distribution, Generic[PosType]):
    def __init__(self,
                 word_distribution: WordDistribution,
                 pos_distribution: CategoricalDistribution[PosType],
                 wordset_size: int):
        self.word_distribution = word_distribution
        self.pos_distribution = pos_distribution
        self.wordset_size = wordset_size
        pos_wordsets: Dict[PosType, List[str]] = {pos: [] for pos in pos_distribution.states}
        empty_pos: Set[PosType] = set(pos_wordsets.keys())
        for i in range(wordset_size):
            if i + len(empty_pos) >= wordset_size:
                break
            word: str = self.word_distribution.generate_sample()
            pos: PosType = self.pos_distribution.generate_sample()
            pos_wordsets[pos].append(word)
            if pos in empty_pos:
                empty_pos.remove(pos)
        for pos in empty_pos:
            word: str = self.word_distribution.generate_sample()
            pos_wordsets[pos].append(word)
        self.pos_word_dists: Dict[PosType, CategoricalDistribution[str]] = {
            pos: CategoricalDistribution.get_random_distribution(states=word_list)
            for pos, word_list in pos_wordsets.items()
        }

    @classmethod
    def get_random_distribution(cls, pos_list: List[PosType], wordset_size=None, **kwargs) -> Self:
        wordset_size: int = wordset_size or PoissonDistribution(500).generate_sample()
        return WordPosDistribution(
            word_distribution=WordDistribution.get_random_distribution(),
            pos_distribution=CategoricalDistribution.get_random_distribution(states=pos_list),
            wordset_size=wordset_size
        )

    def generate_sample(self, pos: PosType, **kwargs) -> str:
        return self.pos_word_dists[pos].generate_sample()

    def derivate(self, distance_scale: float) -> Self:
        raise NotImplementedError()

    def describe(self) -> str:
        pos_dists = '\n'.join([f'{pos}:{to_sub_description(dist.describe())}'
                               for pos, dist in self.pos_word_dists.items()])
        return f'Word Distribution:{to_sub_description(self.word_distribution.describe())}\n' \
               f'Word-Pos Distribution:{to_sub_description(pos_dists)}'


class SentenceDistribution(Distribution, Generic[PosType]):
    def __init__(self,
                 wordpos_dist_distribution: CategoricalDistribution[WordPosDistribution[PosType]],
                 pos_markov_dist: MarkovDistribution[PosType],
                 length_dist: PoissonDistribution):
        self.wordpos_dist_distribution = wordpos_dist_distribution
        self.pos_markov_dist = pos_markov_dist
        self.length_dist = length_dist

    @classmethod
    def get_random_distribution(cls, wordpos_dists: List[WordPosDistribution[PosType]], **kwargs) -> Self:
        pos_intersection = list(set.intersection(*[set(dist.pos_word_dists.keys()) for dist in wordpos_dists]))
        return SentenceDistribution(
            wordpos_dist_distribution=CategoricalDistribution.get_random_distribution(states=wordpos_dists),
            pos_markov_dist=MarkovDistribution.get_random_distribution(states=pos_intersection + ['$']),
            length_dist=PoissonDistribution.get_random_distribution())

    def generate_sample(self, **kwargs) -> str:
        pos_sequence = self.pos_markov_dist.generate_sample(lenght=self.length_dist.generate_sample() + 1)
        word_sequence = [self.wordpos_dist_distribution.generate_sample().generate_sample(pos) for pos in pos_sequence]
        return ' '.join(word_sequence) + '.'

    def derivate(self, distance_scale: float) -> Self:
        raise NotImplementedError()

    def describe(self) -> str:
        return f'Word Count Distribution:{to_sub_description(self.length_dist.describe())}\n' \
               f'Word-Pos Choose Distribution:{to_sub_description(self.wordpos_dist_distribution.describe())}\n' \
               f'Pos Distribution:{to_sub_description(self.pos_markov_dist.describe())}'


TextDistributionType = TypeVar('TextDistributionType')


class IidConcatenatedTextDistribution(Distribution, Generic[TextDistributionType], ABC):
    text_distribution_type: Type[TextDistributionType]
    text_splitter: str

    def __init__(self,
                 text_distribution: Distribution,
                 length_dist: PoissonDistribution
                 ):
        self.text_distribution = text_distribution
        self.length_dist = length_dist

    @classmethod
    def get_random_distribution(cls, **kwargs) -> Self:
        return cls(
            text_distribution=cls.text_distribution_type.get_random_distribution(**kwargs),
            length_dist=PoissonDistribution.get_random_distribution()
        )

    def generate_sample(self, **kwargs) -> str:
        text_count = self.length_dist.generate_sample() + 1
        texts = [self.text_distribution.generate_sample() for _ in range(text_count)]
        return self.text_splitter.join(texts)

    def derivate(self, distance_scale: float) -> Self:
        result = DocumentDistribution(
            text_distribution=self.text_distribution.derivate(distance_scale=distance_scale),
            length_dist=self.text_distribution.derivate(distance_scale=distance_scale)
        )
        return result


class ParagraphDistribution(IidConcatenatedTextDistribution[SentenceDistribution]):
    text_distribution_type = SentenceDistribution
    text_splitter = ' '

    def describe(self) -> str:
        return f'Sentence Count Distribution:{to_sub_description(self.length_dist.describe())}\n' \
               f'Sentence Distribution:{to_sub_description(self.text_distribution.describe())}'


class DocumentDistribution(IidConcatenatedTextDistribution[ParagraphDistribution]):
    text_distribution_type = ParagraphDistribution
    text_splitter = '\n'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children: List[Self] = []

    def derivate(self, *args, **kwargs) -> Self:
        result = super(DocumentDistribution, self).derivate(*args, **kwargs)
        self.children.append(result)
        return result

    def describe(self) -> str:
        return f'Paragraph Count Distribution:{to_sub_description(self.length_dist.describe())}\n' \
               f'Paragraph Distribution:{to_sub_description(self.text_distribution.describe())}' + \
               ''.join([f'\nDerivated #{i}: {to_sub_description(self.children[i].describe())}'
                        for i in range(len(self.children))])
