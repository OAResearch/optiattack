"""Module contains the Archive class, which is a base class for the archives used in the search service."""
from typing import Callable, List, Optional

import numpy as np

from core.search.evaluated_individual import EvaluatedIndividual
from core.search.fitness_value import FitnessValue
from core.search.service.randomness import Randomness
from core.search.service.search_time_controller import SearchTimeController
from core.search.solution import Solution
from core.utils.images import ProcessedImage
from core.utils.nut_request import NutRequest


class Archive:

    """Base class for the archives used in the search service."""

    def __init__(self, stc: SearchTimeController, randomness: Randomness, config: dict) -> None:
        """Initialize the archive."""

        self.original_predication_results = None
        self._populations: List[EvaluatedIndividual] = []
        self._pending: Optional[Callable[[], EvaluatedIndividual]] = None
        self.last_chosen: list[EvaluatedIndividual] = []
        self.stc = stc
        self.randomness = randomness
        self.config = config
        self.image: ProcessedImage = ProcessedImage(None, None, None)
        self.minimized_solution: Optional[Solution] = None

    @property
    def populations(self) -> List[EvaluatedIndividual]:
        """Individuals of the archive; a pending individual (see `set_pending`) is built on this first read."""
        if self._pending is not None:
            builder, self._pending = self._pending, None
            self._populations = [builder()]
        return self._populations

    @populations.setter
    def populations(self, value: List[EvaluatedIndividual]) -> None:
        """Replace the individuals of the archive, discarding a pending individual."""
        self._pending = None
        self._populations = value

    def clean_population(self):
        """Clean the populations list."""
        self.populations = []

    def add_archive_if_needed(self, individual: EvaluatedIndividual,
                              parent: Optional[EvaluatedIndividual] = None) -> bool:
        """Add an individual to the archive if it is better than the current best solution."""
        if self.stc.get_current_fitness_value() > individual.fitness.value:
            self.stc.set_current_fitness(individual.fitness)
            self.populations.append(individual)
            self.stc.new_action_improvement()

            if parent is not None:
                parent.sampling_counter = 0
            return True
        return False

    def replace_with(self, ei: EvaluatedIndividual) -> None:
        """Replace the whole population with a single individual and make its fitness the current one."""
        self.populations = [ei]
        self.stc.set_current_fitness(ei.fitness)
        self.stc.new_action_improvement()

    def set_pending(self, builder: Callable[[], EvaluatedIndividual], fitness: FitnessValue) -> None:
        """
        Lazy `replace_with`: make `fitness` the current one now, but build the individual only when the archive is read.

        `builder` must return an individual with this fitness. It replaces the whole population on the next read of
        `populations` (and so of `get_actions`, `get_mutated_image` or `extract_solution`); a later `set_pending` or
        `replace_with` discards it unbuilt. Used by the baseline, whose new bests can be costly to convert to actions.
        """
        self.stc.set_current_fitness(fitness)
        self.stc.new_action_improvement()
        self._pending = builder

    def sample_individual(self):
        """Sample an individual from the archive."""

        if self.populations.__len__() == 0:
            return

        sampling_counter = np.array([i.sampling_counter for i in self.populations])
        random_index = self.randomness.random_choice(np.flatnonzero(sampling_counter == min(sampling_counter)))
        individual = self.populations[random_index]
        self.last_chosen = individual
        individual.sampling_counter += 1
        return individual

    def shrink_archive(self):
        """Remove individuals from the archive that do not improve the fitness of the current best solution."""
        raise NotImplementedError("This method should be implemented")

    def is_empty(self) -> bool:
        """Check if the archive is empty."""
        return len(self.populations) == 0

    def number_of_population(self) -> int:
        """Return the number of population in the archive."""
        return len(self.populations)

    def set_image(self, image) -> None:
        """Set the image."""
        self.image = image

    def get_image(self) -> ProcessedImage:
        """Get the image."""
        return self.image

    def get_actions(self) -> list:
        """Get the actions of the archive."""
        actions = [ei.individual.get_actions() for ei in self.populations]
        return [action for sublist in actions for action in sublist]

    def get_mutated_image(self, actions=None) -> np.ndarray:
        """Get mutated image. This image contains the actions of the archive."""
        action_image = self.image.array.copy()

        if actions is None:
            actions = self.get_actions()

        for action in actions:
            x, y = action.get_location()
            action_image[x, y] = action.get_color()
        return action_image

    def set_original_prediction_results(self, results):
        """Set the original prediction results."""
        self.original_predication_results = NutRequest(results, self.config.get("target"))

    def get_original_prediction_results(self):
        """Get the original prediction results."""
        return self.original_predication_results

    def set_minimized_solution(self, solution):
        """Set the minimized solution."""
        self.minimized_solution = solution

    def extract_solution(self, not_minimized=False) -> Solution:
        """Extract the solution from the archive. If the solution is already minimized, return it."""

        if self.minimized_solution is not None and not not_minimized:
            return self.minimized_solution

        # Later actions overwrite earlier ones at the same location (see get_mutated_image),
        # so keep only the last action per location, in order of their last occurrence.
        seen_locations = set()
        actions = []
        for action in reversed(self.get_actions()):
            location = action.get_location()
            if location not in seen_locations:
                seen_locations.add(location)
                actions.append(action)
        actions.reverse()

        return Solution(actions, self.stc.get_current_fitness())
