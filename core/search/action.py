"""Represents an action with a location, RGB color, noise calculation, and parent-child relationship."""
from typing import Tuple
import numpy as np


class Action:

    """Represents an action with a location, RGB color, noise calculation, and parent-child relationship."""

    def __init__(self, location: Tuple[int, int], red: int, green: int, blue: int) -> None:
        """Initializes an Action instance with a location, RGB color values, and default parent (None) and noise (0)."""
        self.location = location
        self.red = red
        self.green = green
        self.blue = blue
        self.parent = None
        self.noise = 0

    def set_parent(self, parent) -> None:
        """Sets the parent of the current action to the provided parent object."""
        self.parent = parent

    def get_parent(self):
        """Returns the parent of the current action."""
        return self.parent

    def get_location(self):
        """Returns the location of the action as a tuple (x, y)."""
        return self.location

    def color(self):
        """Returns the RGB color of the action as a NumPy array [red, green, blue]."""
        return np.array([self.red, self.green, self.blue])

    def calculate_noise(self, current_value):
        """
        Calculates and sets the noise.

        Calculates and sets the noise as the sum of absolute differences between the
        action's color and a provided current_value.
        """

        self.noise = np.sum(np.abs(self.color() - current_value))
        return self.noise

    def __eq__(self, o: object) -> bool:
        """Compares two Action objects for equality based on their location and color."""
        return self.location == o.location and all(self.color() == o.color())

    def __str__(self) -> str:
        """Returns a string representation of the action, including its location and color."""
        return f"Action at {self.location} with color {self.color()}"
