"""Targeted implementation of the FitnessFunction abstract class."""

from typing import Optional, TypeVar

from core.search.service.fitness_function.fitness_function import FitnessFunction
from core.search.action import Action
from core.search.fitness_value import FitnessValue
from core.search.individual import Individual

T = TypeVar('T', bound=Individual)


class TargetedFitnessFunction(FitnessFunction):

    """Targeted fitness function that attempts to change classification to a specific target."""

    def evaluate(self, individual: Optional[T] = None, actions: Optional[list[Action]] = None) -> FitnessValue:
        """Targeted fitness function that attempts to change classification to a specific target."""
        raise NotImplementedError("Targeted fitness function is not yet implemented.")
