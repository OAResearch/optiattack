"""Single Point Crossover Implementation"""

import logging

from core.search.service.crossover.crossover import Crossover


class SinglePointCrossover(Crossover):

    """Single point crossover implementation for genetic algorithms."""

    def apply_crossover(self, parent1, parent2):
        """Applies crossover between two parents to create a new individual."""
        p1 = parent1.copy()
        p2 = parent2.copy()

        if p1.get_actions().__len__() < 2 or p2.get_actions().__len__() < 2:
            logging.debug("Crossover not applied due to insufficient actions in parents.")
            return parent1, parent2

        split_point = self.randomness.next_float()

        pos1 = int((len(p1.get_actions()) - 1) * split_point) + 1
        pos2 = int((len(p2.get_actions()) - 1) * split_point) + 1

        parent1.crossover(p2, pos1, pos2)
        parent2.crossover(p1, pos2, pos1)
