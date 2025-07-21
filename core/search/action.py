"""Represents an action with a location, RGB color, noise calculation, and parent-child relationship."""
from typing import Tuple
import numpy as np
from numpy import ndarray


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

    def set_location(self, x, y):
        """Sets the location of the action to the provided x and y values."""
        self.location = (x, y)

    def get_color(self):
        """Returns the RGB color of the action as a NumPy array [red, green, blue]."""
        return np.array([self.red, self.green, self.blue])

    def set_color(self, color):
        """Sets the RGB color of the action to the provided color."""
        self.red, self.green, self.blue = color

    def calculate_noise(self, current_value):
        """
        Calculates and sets the noise.

        Calculates and sets the noise as the sum of absolute differences between the
        action's color and a provided current_value.
        """

        self.noise = np.sum(np.abs(self.get_color() - current_value))
        return self.noise

    def __eq__(self, o: object) -> bool:
        """Compares two Action objects for equality based on their location and color."""
        if not isinstance(o, Action):
            return False
        return self.location == o.location and all(self.get_color() == o.get_color())

    def same_location(self, o: object) -> bool:
        """Compares two Action objects for equality based on their location."""
        if not isinstance(o, Action):
            return False
        return self.location == o.location

    def __hash__(self) -> int:
        """Returns a hash of the Action object based on its location and color."""
        return hash((self.location, tuple(self.get_color())))

    def __str__(self) -> str:
        """Returns a string representation of the action, including its location and color."""
        return f"Action at {self.location} with color {self.get_color()}"

    def copy(self):
        """Returns a copy of the action."""
        action = Action(self.location, self.red, self.green, self.blue)
        action.set_parent(self.parent)
        action.noise = self.noise
        return action

    def to_array(self):
        """Returns the action's color as a NumPy array."""
        return [self.location[0], self.location[1], self.get_color(), 0]