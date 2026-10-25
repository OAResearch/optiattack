"""Mutator class is the base class for all mutators. It provides the basic structure for all mutators."""
import numpy as np

from core.search.individual import Individual
from core.search.service.adaptive_parameter_control import AdaptiveParameterControl
from core.search.service.archive import Archive
from core.search.service.randomness import Randomness
from core.search.service.search_time_controller import SearchTimeController
from typing import Optional


class Mutator:

    """Mutator class is the base class for all mutators. It provides the basic structure for all mutators."""

    def __init__(self, randomness: Randomness,
                 stc: SearchTimeController,
                 config: dict,
                 apc: AdaptiveParameterControl,
                 archive: Optional[Archive] = None):
        """Initializes the mutator with the randomness, time controller, and configuration."""
        self.randomness = randomness
        self.stc = stc
        self.config = config
        self.apc = apc
        self.archive = archive

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
