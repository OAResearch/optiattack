from core.search.action import Action
from core.search.fitness_value import FitnessValue


class Solution:
    def __init__(self, actions: list[Action], fitness_value: FitnessValue):
        self.actions = actions
        self.fitness_value = fitness_value
