"""Module defines the FitnessFunction class, which represents a fitness function that can be used to evaluate the"""

from typing import TypeVar, Optional

from core.remote.remote_controller import RemoteController
from core.search.action import Action
from core.search.evaluated_individual import EvaluatedIndividual
from core.search.fitness_value import FitnessValue
from core.search.individual import Individual
from core.search.service.archive import Archive
from core.search.service.search_time_controller import SearchTimeController

T = TypeVar('T', bound=Individual)


class FitnessFunction:

    """Represents a fitness function that can be used to evaluate the fitness of individuals."""

    def __init__(self, archive: Archive, remote_controller: RemoteController, stc: SearchTimeController) -> None:
        """Initializes a FitnessFunction instance with the provided archive."""
        self.archive = archive
        self.remote_controller = remote_controller
        self.stc = stc

    """Represents a fitness function that can be used to evaluate the fitness of individuals."""

    def evaluate(self, individual: Optional[T] = None, actions: Optional[list[Action]] = None) -> FitnessValue:
        """Evaluates the fitness of the provided individual and returns a float value."""
        img_array = self.archive.get_mutated_image(actions)

        if individual is not None:
            img_array = individual.get_action_image(img_array)

        result = self.remote_controller.new_action(img_array)
        original_result = self.archive.get_original_prediction_results()
        fitness_value = result.max_score.value - result.second_max_score.value \
            if result.max_score.label == original_result.max_score.label else 0.0
        return FitnessValue(fitness_value, result.predictions)

    def calculate_fitness(self, individual: T) -> EvaluatedIndividual:
        """Calculate the fitness of an individual."""

        fitness_value = SearchTimeController.measure_time_millis(
            self.log_execution_time,
            lambda: self.evaluate(individual=individual),
            individual.actions.__len__()
        )
        ei = EvaluatedIndividual(individual, fitness_value)
        return ei

    def calculate_fitness_with_actions(self, actions: list[Action]) -> FitnessValue:
        """Calculate the fitness of an individual."""
        fitness_value = self.evaluate(actions=actions)
        return fitness_value

    def log_execution_time(self, t: int, ind: FitnessValue, action_size: int = 1) -> None:
        """Log the execution time and update the individual's execution time."""
        self.stc.report_executed_individual_time(t, action_size)
        ind.set_execution_time_ms(t)
