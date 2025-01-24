"""Mutator class is the base class for all mutators. It provides the basic structure for all mutators."""

from core.search.individual import Individual
from core.search.service.archive import Archive
from core.search.service.randomness import Randomness
from core.search.service.search_time_controller import SearchTimeController


class Mutator:

    """Mutator class is the base class for all mutators. It provides the basic structure for all mutators."""

    def __init__(self, randomness: Randomness,
                 stc: SearchTimeController,
                 config: dict):
        """Initializes the mutator with the randomness, time controller, and configuration."""
        self.randomness = randomness
        self.stc = stc
        self.config = config

    def mutate(self, individual: Individual):
        """Mutates the individual."""
        raise NotImplementedError("Mutate method must be implemented in subclass.")

    def mutate_and_save(self, individual: Individual, archive: Archive):
        """Mutates the individual and saves it to the archive."""
        raise NotImplementedError("Mutate method must be implemented in subclass.")

    def check_limit_values(self, value):
        """Checks the limit values of the value."""
        value[value < 0] = 0
        value[value > 255] = 255
        return value
