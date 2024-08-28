from typing import Any, Type

from .base import Distribution


class TextDistribution(Distribution):
    class ParameterHolder(Distribution.ParameterHolder):

        @classmethod
        def get_random_param_holder(cls) -> Distribution.ParameterHolder:
            pass

        def derivate_param_holder(self, distance: float) -> Distribution.ParameterHolder:
            pass

    @classmethod
    def _get_param_holder_class(cls) -> Type[Distribution.ParameterHolder]:
        pass

    def generate_sample(self) -> Any:
        pass
