"""Untargeted implementation of the FitnessFunction abstract class."""

from typing import Optional, TypeVar

from core.search.action import Action
from core.search.fitness_value import FitnessValue
from core.search.individual import Individual
from core.search.service.fitness_function.fitness_function import FitnessFunction

T = TypeVar('T', bound=Individual)


class UntargetedFitnessFunction(FitnessFunction):

    """Untargeted fitness function that attempts to change classification without any target."""

    def evaluate(self, individual: Optional[T] = None, actions: Optional[list[Action]] = None) -> FitnessValue:
        """Untargeted fitness function that attempts to change classification without any target."""

        img_array = self.archive.get_mutated_image(actions)

        if individual is not None:
            img_array = individual.get_action_image(img_array)

        result = self.remote_controller.new_action(img_array)
        original_result = self.archive.get_original_prediction_results()

        fitness_value = result.max_score.value - result.second_max_score.value \
            if result.max_score.label == original_result.max_score.label \
            else result.second_max_score.value - result.max_score.value

        return FitnessValue(fitness_value, result.predictions)
