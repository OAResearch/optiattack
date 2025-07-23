"""
File is part of the following publication:

Bartlett, A., Liem, C. C., & Panichella, A. (2024).
Multi-objective differential evolution in the generation of adversarial examples.
Science of Computer Programming, 238, 103169.
"""

# src/eliminations.py

from pymoo.core.duplicate import ElementwiseDuplicateElimination
import numpy as np


# SimpleElimination - Removes any duplicate offspring.
class SimpleElimination(ElementwiseDuplicateElimination):

    """Class implements a simple elimination strategy that checks for duplicate individuals"""

    def is_equal(self, a, b):
        """Check if two individuals are equal based on their first element in X."""
        return np.array_equal(a.X[0], b.X[0])
