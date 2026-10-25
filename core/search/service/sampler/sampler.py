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

    def check_location_limits(self, row, col):
        """Checks the limit values of the location. Locations are (row, col) indices of the image array."""
        max_row = self.config.get("image_height")
        max_col = self.config.get("image_width")

        if row < 0:
            row = 0
        if row >= max_row:
            row = max_row - 1
        if col < 0:
            col = 0
        if col >= max_col:
            col = max_col - 1

        return int(row), int(col)
