from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Type


class Distribution(ABC):
    class ParameterHolder(ABC):
        @classmethod
        @abstractmethod
        def get_random_param_holder(cls) -> Distribution.ParameterHolder:
            raise NotImplementedError()

        @abstractmethod
        def derivate_param_holder(self, distance: float) -> Distribution.ParameterHolder:
            raise NotImplementedError()

    def __init__(self, params: ParameterHolder):
        self._params = params

    @classmethod
    @abstractmethod
    def _get_param_holder_class(cls) -> Type[Distribution.ParameterHolder]:
        raise NotImplementedError()

    @classmethod
    def get_random_distribution(cls) -> Distribution:
        return cls(cls._get_param_holder_class().get_random_param_holder())

    @abstractmethod
    def generate_sample(self) -> Any:
        raise NotImplementedError()

    def generate_child(self, distance: float) -> Distribution:
        return self.__class__(self._params.derivate_param_holder(distance))
