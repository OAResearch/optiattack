"""Module contains the Archive class, which is a base class for the archives used in the search service."""
from typing import Optional

import numpy as np

from core.search.evaluated_individual import EvaluatedIndividual
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
        self.populations: list[EvaluatedIndividual] = []
        self.last_chosen: list[EvaluatedIndividual] = []
        self.stc = stc
        self.randomness = randomness
        self.config = config
        self.image: ProcessedImage = ProcessedImage(None, None, None)

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

    def get_mutated_image(self) -> np.ndarray:
        """Get mutated image. This image contains the actions of the archive."""
        action_image = self.image.array.copy()
        actions = self.get_actions()
        for action in actions:
            x, y = action.get_location()
            action_image[x, y] = action.get_color()
        return action_image

    def set_original_prediction_results(self, results):
        """Set the original prediction results."""
        self.original_predication_results = NutRequest(results)

    def get_original_prediction_results(self):
        """Get the original prediction results."""
        return self.original_predication_results

    def extract_solution(self) -> Solution:
        """Extract the solution from the archive."""
        actions = self.get_actions().copy()

        for action in actions:
            location = action.get_location()
            number_of_locations = len([a for a in actions if a.get_location() == location])
            if number_of_locations > 1:
                actions.remove(action)

        return Solution(actions, self.stc.get_current_fitness())
