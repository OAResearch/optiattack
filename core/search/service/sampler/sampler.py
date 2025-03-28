"""Abstract class for sampling individuals from the search space."""

from core.search.individual import Individual
from core.search.service.randomness import Randomness


class Sampler:

    """Abstract class for sampling individuals from the search space."""

    def __init__(self, randomness: Randomness, config: dict):
        """Initialize the sampler."""

        self.randomness = randomness
        self.config = config

    def sample(self) -> Individual:
        """Sample an individual."""
        raise NotImplementedError("This method should be implemented")
