"""Targeted implementation of the FitnessFunction abstract class."""

from typing import Optional, TypeVar

from core.search.action import Action
from core.search.fitness_value import FitnessValue
from core.search.individual import Individual
from core.search.service.fitness_function.fitness_function import FitnessFunction

T = TypeVar('T', bound=Individual)


class TargetedFitnessFunction(FitnessFunction):

    """Targeted fitness function that attempts to change classification to a specific target."""

    def __init__(self, archive, remote_controller, stc, target):
        """Initialize the targeted fitness function with a model."""
        super().__init__(archive, remote_controller, stc)
        self.target = target

    def evaluate(self, individual: Optional[T] = None, actions: Optional[list[Action]] = None) -> FitnessValue:
        """Targeted fitness function that attempts to change classification to the target class."""

        img_array = self.archive.get_mutated_image(actions)

        if individual is not None:
            img_array = individual.get_action_image(img_array)

        result = self.remote_controller.new_action(img_array)
        original_result = self.archive.get_original_prediction_results()

        fitness_value = result.max_score.value - result.targeted_score.value \
            if result.max_score.label == original_result.max_score.label else 0.0

        return FitnessValue(fitness_value, result.predictions)
