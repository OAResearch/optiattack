"""Class used to keep track of passing of time during the search."""

import logging
import time
from collections import deque
from typing import Tuple, TypeVar, Callable

from core.config_parser import ConfigParser
from core.search.fitness_value import FitnessValue
from core.search.phase_controller import PhaseController
from core.utils.incremental_average import IncrementalAverage


class SearchTimeController:

    """Class used to keep track of passing of time during the search."""

    def __init__(self, config, pc: PhaseController):
        """Initialize the search time controller."""
        self.config = config
        self.pc = pc
        self.logger = logging.getLogger(__name__)
        self.evaluated_individuals = 0
        self.evaluated_individuals_in_pruning = 0
        self.last_action_improvement_timestamp = 0
        self.start_time = 0
        self.search_started = False
        self.average_test_time_ms = IncrementalAverage()
        self.executed_individual_time: deque[Tuple[int, int]] = deque(maxlen=100)
        self.current_fitness_value = FitnessValue(1.0, list())
        self.pruned_fitness_value = FitnessValue(1.0, list())

        self.listeners = []

    # Generic type for the function return value
    T = TypeVar("T")

    @staticmethod
    def measure_time_millis(logging_function: Callable[[int, T, int], None],
                            function: Callable[[], T], action_size: int = 1) -> T:
        """Measure the execution time of a function and log the result using the provided logging function."""
        start_time = int(time.time() * 1000)  # Current time in milliseconds
        result = function()  # Execute the function
        elapsed_time = int(time.time() * 1000) - start_time  # Calculate elapsed time in milliseconds

        # Call the logging function with the elapsed time and result
        logging_function(elapsed_time, result, action_size)

        return result

    def get_current_fitness(self):
        """Get the best individual found so far."""
        return self.current_fitness_value

    def get_current_fitness_value(self):
        """Get the best individual found so far."""
        return self.current_fitness_value.value

    def set_current_fitness(self, value):
        """Set the best individual found so far."""
        self.current_fitness_value = value

    def get_pruned_fitness(self):
        """Get the best individual found so far."""
        return self.pruned_fitness_value

    def get_pruned_fitness_value(self):
        """Get the best individual found so far."""
        return self.pruned_fitness_value.value

    def set_pruned_fitness(self, value):
        """Set the best individual found so far."""
        self.pruned_fitness_value = value

    def start_search(self):
        """Start the search time controller"""
        self.start_time = time.time()
        self.search_started = True
        self.last_action_improvement_timestamp = self.start_time

    def new_individual_evaluation(self):
        """Update the number of evaluated individuals."""
        if self.pc.is_pruning():
            self.evaluated_individuals_in_pruning += 1
            return

        self.evaluated_individuals += 1

        for listener in self.listeners:
            listener.new_action_evaluated()

    def get_evaluated_individuals(self):
        """Get the number of evaluated individuals."""
        return self.evaluated_individuals

    def new_action_improvement(self):
        """Update the timestamp of the last action improvement."""
        self.last_action_improvement_timestamp = int(time.time() * 1000)

    def get_elapsed_seconds(self):
        """Get the elapsed time in seconds."""
        return time.time() - self.start_time

    def get_elapsed_time(self):
        """Get the elapsed time in HH:MM:SS format."""
        return time.strftime('%H:%M:%S', time.gmtime(self.get_elapsed_seconds()))

    def percentage_used_budget(self):
        """Get the percentage of the budget used."""
        if not self.search_started:
            return 0.0

        if self.config.get('stopping_criterion') == ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS:
            return self.get_evaluated_individuals() / self.config.get('max_evaluations')

        elif self.config.get('stopping_criterion') == ConfigParser.StoppingCriterion.TIME:
            return self.get_elapsed_seconds() / self.config.get('max_evaluations')

        else:
            raise ValueError("Not supported stopping criterion")

    def should_continue_search(self):
        """Check if the search should continue."""
        return self.percentage_used_budget() < 1.0 and self.get_current_fitness_value() > 0.0

    def add_listener(self, listener):
        """Add a listener to the search time controller."""
        self.listeners.append(listener)

    def report_executed_individual_time(self, ms: int, n_actions: int):
        """Report the execution time of an individual test and the number of actions."""
        # if not self.recording:
        #     return

        # Add the new data to the queue (automatically removes oldest if maxlen is exceeded)
        self.executed_individual_time.append((ms, n_actions))

        # Update the incremental averages
        self.average_test_time_ms.add_value(ms)

    def compute_executed_individual_time_statistics(self) -> Tuple[float, float]:
        """Compute the average execution time and average number of actions for the last 100 tests."""
        if not self.executed_individual_time:
            return 0.0, 0.0

        # Calculate the average execution time and average number of actions
        avg_ms = sum(ms for ms, _ in self.executed_individual_time) / len(self.executed_individual_time)
        avg_actions = sum(actions for _, actions in self.executed_individual_time) / len(self.executed_individual_time)

        return avg_ms, avg_actions

    def get_seconds_since_last_improvement(self) -> int:
        """Getter for the seconds since last improvement"""
        current_time_ms = int(time.time() * 1000)
        elapsed_time_seconds = (current_time_ms - self.last_action_improvement_timestamp) / 1000.0
        return int(elapsed_time_seconds)
