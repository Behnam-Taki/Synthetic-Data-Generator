from typing import Any

from .base import Distribution


class WordDistribution(Distribution):
    @classmethod
    def get_random_distribution(cls) -> Distribution:
        pass

    def generate_sample(self) -> Any:
        pass

    def generate_child(self, distance: float) -> Distribution:
        pass
