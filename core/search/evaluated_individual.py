"""Represents an evaluated individual with its fitness value and the individual itself."""
from core.search.fitness_value import FitnessValue
from core.search.individual import Individual


class EvaluatedIndividual:

    """Represents an evaluated individual with its fitness value and the individual itself."""

    def __init__(self, individual: Individual, fitness: FitnessValue):
        """Initializes EvaluatedIndividual with the provided fitness value and individual."""

        self.fitness = fitness
        self.individual = individual
        self.sampling_counter = 0

    def copy(self):
        """Returns a copy of the EvaluatedIndividual."""
        return EvaluatedIndividual(self.individual.copy(), self.fitness.copy())
