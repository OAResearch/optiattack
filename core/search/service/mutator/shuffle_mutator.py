"""Shuffle mutation implementation for genetic algorithms."""

import numpy as np
from core.search.individual import Individual
from core.search.service.mutator.mutator import Mutator
from core.search.service.shuffle_block import ShuffleBlockMixin


class ShuffleMutator(ShuffleBlockMixin, Mutator):

    """Shuffle mutator - block shuffling with Gaussian shift."""

    def mutate(self, individual: Individual) -> Individual:
        """Apply Gaussian shift to block center and reshuffle."""
        if len(individual.get_actions()) == 0:
            return self.add_shuffle_mutation()

        # Find block center from existing actions
        actions = individual.get_actions()
        locations = [a.get_location() for a in actions]
        rows = [loc[0] for loc in locations]
        cols = [loc[1] for loc in locations]

        if not cols or not rows:
            return self.add_shuffle_mutation()

        row = int(np.median(rows))
        col = int(np.median(cols))

        # Gaussian shift
        sigma = self.apc.get_location_apc()
        row = int(np.round(self.randomness.random_gaussian(row, sigma)))
        col = int(np.round(self.randomness.random_gaussian(col, sigma)))

        # Clamp to valid bounds
        row, col = self._clamp_position(row, col)

        if self.archive is None:
            raise ValueError("Archive is required for shuffle mutation")

        # Extract, shuffle, and create new individual
        seed = self.archive.image.array.copy()
        block = self._extract_block(seed, row, col)
        is_float = self._is_float_image(block)
        shuffled_block = self._shuffle_block(block)

        new_individual = Individual()
        self._create_actions_from_block(new_individual, shuffled_block, row, col, is_float)

        return new_individual

    def add_shuffle_mutation(self) -> Individual:
        """Create initial shuffled block at random non-colliding position."""
        return self.sample_shuffled_block()
