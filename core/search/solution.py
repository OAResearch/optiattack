"""Solution class for the whole search process."""
from core.search.action import Action
from core.search.fitness_value import FitnessValue


class Solution:

    """Solution class for the whole search process."""

    def __init__(self, actions: list[Action], fitness_value: FitnessValue):
        """Initialize the solution."""
        self.actions = actions
        self.fitness_value = fitness_value
