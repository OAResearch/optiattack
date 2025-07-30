"""Crossover class is the base class for all crossovers in the search service."""

from core.search.individual import Individual
from core.search.service.adaptive_parameter_control import AdaptiveParameterControl
from core.search.service.randomness import Randomness
from core.search.service.search_time_controller import SearchTimeController


class Crossover:

    """Crossover class is the base class for all crossovers. It provides the basic structure for all crossovers."""

    def __init__(self, randomness: Randomness,
                 stc: SearchTimeController,
                 config: dict,
                 apc: AdaptiveParameterControl):
        """Initializes the crossover with the randomness, time controller, and configuration."""
        self.randomness = randomness
        self.stc = stc
        self.config = config
        self.apc = apc

    def apply_crossover(self, parent1: Individual, parent2: Individual) -> None:
        """Applies crossover between two parents to create a new individual."""
        raise NotImplementedError("Crossover method must be implemented in subclass.")
