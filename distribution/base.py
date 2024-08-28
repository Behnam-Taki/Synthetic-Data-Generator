from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Type


class Distribution(ABC):
    @classmethod
    @abstractmethod
    def get_random_distribution(cls) -> Distribution:
        raise NotImplementedError()

    @abstractmethod
    def generate_sample(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def generate_child(self, distance: float) -> Distribution:
        raise NotImplementedError()
