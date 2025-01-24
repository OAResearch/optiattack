"""Represents an evaluated individual with its fitness value and the individual itself."""


class EvaluatedIndividual:

    """Represents an evaluated individual with its fitness value and the individual itself."""

    def __init__(self, individual, fitness):
        """Initializes EvaluatedIndividual with the provided fitness value and individual."""

        self.fitness = fitness
        self.individual = individual
