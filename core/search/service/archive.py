"""Module contains the Archive class, which is a base class for the archives used in the search service."""

from core.search.action import Action
from core.search.evaluated_individual import EvaluatedIndividual
from core.search.individual import Individual
from core.search.service.randomness import Randomness
from core.search.service.search_time_controller import SearchTimeController
from core.utils.images import ProcessedImage
from core.utils.nut_request import NutRequest


class Archive:

    """Base class for the archives used in the search service."""

    def __init__(self, stc: SearchTimeController, randomness: Randomness, config: dict):
        """Initialize the archive."""

        self.original_predication_results = None
        self.populations: list[EvaluatedIndividual] = []
        self.sampling_counter = 0
        self.last_chosen: list[EvaluatedIndividual] = []
        self.stc = stc
        self.randomness = randomness
        self.config = config
        self.image: ProcessedImage = ProcessedImage(None, None, None)

    def clean_population(self):
        """Clean the populations list."""
        self.populations = []

    def add_archive_if_needed(self, individual: EvaluatedIndividual):
        """Add an individual to the archive if it is better than the current best solution."""
        if self.stc.get_current_fitness() > individual.fitness.value:
            self.stc.set_current_fitness(individual.fitness.value)
            self.populations.append(individual)
            self.stc.new_action_improvement()

    def shrink_archive(self):
        """Remove individuals from the archive that do not improve the fitness of the current best solution."""
        raise NotImplementedError("This method should be implemented")

    def sample_individual(self):
        """Sample an individual randomly."""
        individual = Individual()
        number_of_actions = self.randomness.next_int(1, 10)
        for _ in range(number_of_actions):
            action = self.sample_random_action()
            while not individual.add_action(action):
                action = self.sample_random_action()
                individual.add_action(action)
        return individual

    def sample_random_action(self):
        """Return a random action."""
        location = (self.randomness.next_int(0, self.config.get("image_width")),
                    self.randomness.next_int(0, self.config.get("image_height")))
        color = self.randomness.next_int(0, 255), self.randomness.next_int(0, 255), self.randomness.next_int(0, 255)
        return Action(location, color[0], color[1], color[2])

    def is_empty(self):
        """Check if the archive is empty."""
        return len(self.populations) == 0

    def number_of_population(self):
        """Return the number of population in the archive."""
        return len(self.populations)

    def set_image(self, image):
        """Set the image."""
        self.image = image

    def get_image(self):
        """Get the image."""
        return self.image

    def get_actions(self):
        """Get the actions of the archive."""
        actions = [ei.individual.get_actions() for ei in self.populations]
        return [action for sublist in actions for action in sublist]

    def get_mutated_image(self):
        """Get mutated image. This image contains the actions of the archive."""
        action_image = self.image.array.copy()
        actions = self.get_actions()
        for action in actions:
            x, y = action.get_location()
            action_image[y, x] = action.get_color()
        return action_image

    def set_original_prediction_results(self, results):
        """Set the original prediction results."""
        self.original_predication_results = NutRequest(results)

    def get_original_prediction_results(self):
        """Get the original prediction results."""
        return self.original_predication_results
