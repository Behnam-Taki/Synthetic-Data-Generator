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

    @staticmethod
    def _to_sub_description(description: str, format_for_one_line: bool = False) -> str:
        return ('\n' + description).replace('\n', '\n   ') \
            if '\n' in description or format_for_one_line \
            else ' ' + description

    def __repr__(self):
        return self.describe()
