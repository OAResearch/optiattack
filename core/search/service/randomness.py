"""A module for generating random numbers and performing various random operations."""

import numpy as np


class Randomness:

    """A class to generate random numbers and perform various random operations."""

    def __init__(self, config):
        """Initializes the Randomness object and sets the random seed."""
        self.random = np.random
        self.update_seed(config.get("seed"))

    def update_seed(self, seed):
        """
        Updates the seed of the random number generator.

        If seed is negative, the generator uses the current CPU time.

        """
        if seed < 0:
            self.random.seed()
        else:
            self.random.seed(seed)

    def next_float(self, min_value=None, max_value=None):
        """
        Generates a random float value between specified bounds.

        Args
        -------
            min_value (float, optional): The minimum bound. Defaults to None.
                If None, defaults to 0.
            max_value (float, optional): The maximum bound. Defaults to None.
                If None, defaults to 1.

        Returns
        -------
            float - A random float between the specified bounds.

        """
        if min_value is None and max_value is None:
            return self.random.uniform(0, 1)

        if min_value is None:
            return self.random.uniform(0, max_value)

        if max_value is None:
            return self.random.uniform(min_value, 1)

        return self.random.uniform(min_value, max_value)

    def next_bool(self, value):
        """
        Generates a boolean value based on a probability.

        Args
        -------
            value (float): A probability (0 <= value <= 1) where True is returned.

        Returns
        -------
            bool: True with probability `value`, otherwise False.

        """
        return self.random.rand() < value

    def next_int(self, min_value=None, max_value=None):
        """
        Generates a random integer between specified bounds.

        Args
        -------
            min_value (int, optional): The minimum bound. Defaults to None.
                If None, defaults to 0.
            max_value (int, optional): The maximum bound. Defaults to None.
                If None, defaults to 1.

        Returns
        -------
            int: A random integer between the specified bounds.

        """
        if min_value is None and max_value is None:
            return self.random.randint(0, 1)

        if min_value is None:
            return self.random.randint(0, max_value)

        if max_value is None:
            return self.random.randint(min_value, 1)

        return self.random.randint(min_value, max_value)

    def random_choice(self, max_size, selection_probs=None):
        """
        Returns a random choice from a range or based on a probability distribution.

        Args
        -------
            max_size (int): The size of the range to select from.
            selection_probs (list, optional): A list of probabilities for each choice.
                Defaults to None, meaning equal probability for all choices.

        Returns
        -------
            int: A randomly selected element from the range.

        """
        if selection_probs is None:
            return self.random.choice(max_size)
        return self.random.choice(max_size, p=selection_probs)

    def random_gaussian(self, mean, std):
        """
        Generates a random float value from a Gaussian distribution.

        Args
        -------
            mean (float): The mean of the Gaussian distribution.
            std (float): The standard deviation of the Gaussian distribution.

        Returns
        -------
            float: A random value from the Gaussian distribution.

        """
        return self.random.normal(mean, std)
