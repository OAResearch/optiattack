"""Abstract class for sampling individuals from the search space."""
import numpy as np

from core.search.individual import Individual
from core.search.service.archive import Archive
from core.search.service.randomness import Randomness


class Sampler:

    """Abstract class for sampling individuals from the search space."""

    def __init__(self, randomness: Randomness, archive: Archive, config: dict):
        """Initialize the sampler."""

        self.randomness = randomness
        self.archive = archive
        self.config = config

    def sample(self) -> Individual:
        """Sample an individual."""
        raise NotImplementedError("This method should be implemented")

    def check_limit_values(self, value):
        """Checks the limit values of the value."""
        value[value < 0] = 0
        value[value > 255] = 255
        # convert to integer
        return np.round(value).astype(int)

    def check_location_limits(self, width, height):
        """Checks the limit values of the location."""
        max_height = self.config.get("image_height")
        max_width = self.config.get("image_width")

        if height < 0:
            height = 0
        if height >= max_height:
            height = max_height - 1
        if width < 0:
            width = 0
        if width >= max_width:
            width = max_width - 1

        return int(width), int(height)
