"""Module defines the FitnessFunction class, which represents a fitness function that can be used to evaluate the"""

from typing import TypeVar

from core.remote.remote_controller import RemoteController
from core.search.fitness_value import FitnessValue
from core.search.individual import Individual
from core.search.service.archive import Archive

T = TypeVar('T', bound=Individual)


class FitnessFunction:

    """Represents a fitness function that can be used to evaluate the fitness of individuals."""

    def __init__(self, archive: Archive, remote_controller: RemoteController) -> None:
        """Initializes a FitnessFunction instance with the provided archive."""
        self.archive = archive
        self.remote_controller = remote_controller

    """Represents a fitness function that can be used to evaluate the fitness of individuals."""

    def evaluate(self, individual: T) -> FitnessValue:
        """Evaluates the fitness of the provided individual and returns a float value."""
        img_array = individual.get_action_image(self.archive.get_image())
        result = self.remote_controller.new_action(img_array)
        original_result = self.archive.get_original_prediction_results()
        fitness_value = result.max_score.value - result.second_max_score.value \
            if result.max_score.label == original_result.max_score.label else 0
        return FitnessValue(fitness_value)
