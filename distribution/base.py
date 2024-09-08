from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Self


class Distribution(ABC):
    @classmethod
    @abstractmethod
    def get_random_distribution(cls, **kwargs) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def generate_sample(self, **kwargs) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def generate_child(self, distance: float) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError()

    def __repr__(self):
        return self.describe()
