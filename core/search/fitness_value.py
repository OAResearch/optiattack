"""
Represents a fitness value and execution time

Represents a fitness value and execution time, with methods to set fitness value
and execution time in milliseconds.
"""


class FitnessValue:

    """
    Represents a fitness value and execution time

    Represents a fitness value and execution time, with methods to set fitness value
    and execution time in milliseconds.
    """

    def __init__(self, value: float, predictions: list):
        """Initializes FitnessValue with default fitness_value=0 and execution_time_ms=0."""
        self.value = value
        self.execution_time_ms = 0
        self.predictions = predictions

    def set_fitness_value(self, value):
        """Sets the fitness value to the provided integer."""
        self.value = value

    def set_execution_time_ms(self, execution_time_ms):
        """Sets the execution time in milliseconds to the provided integer."""
        self.execution_time_ms = execution_time_ms

    def set_predictions(self, predictions):
        """Sets the predictions to the provided list."""
        self.predictions = predictions
