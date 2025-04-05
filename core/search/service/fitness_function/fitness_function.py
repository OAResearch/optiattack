"""Abstract class for fitness functions that evaluate individuals in the search space."""

from typing import TypeVar, Optional

from core.search.action import Action
from core.search.evaluated_individual import EvaluatedIndividual
from core.search.fitness_value import FitnessValue
from core.search.individual import Individual
from core.search.service.archive import Archive
from core.search.service.search_time_controller import SearchTimeController
from core.remote.remote_controller import RemoteController

T = TypeVar('T', bound=Individual)


class FitnessFunction:
    """Abstract class for fitness functions that evaluate individuals in the search space."""

    def __init__(self, archive: Archive, remote_controller: RemoteController, stc: SearchTimeController) -> None:
        """Initialize the fitness function."""
        self.archive = archive
        self.remote_controller = remote_controller
        self.stc = stc

    def evaluate(self, individual: Optional[T] = None, actions: Optional[list[Action]] = None) -> FitnessValue:
        """Evaluate the fitness of the provided individual and return a fitness value.

        This method should be implemented by concrete subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

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
        """Calculate the fitness of a list of actions."""
        fitness_value = self.evaluate(actions=actions)
        return fitness_value

    def log_execution_time(self, t: int, ind: FitnessValue, action_size: int = 1) -> None:
        """Log the execution time and update the individual's execution time."""
        self.stc.report_executed_individual_time(t, action_size)
        ind.set_execution_time_ms(t)