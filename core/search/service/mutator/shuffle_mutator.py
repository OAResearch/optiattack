"""Shuffle mutation implementation for genetic algorithms."""

import numpy as np
from core.search.action import Action
from core.search.individual import Individual
from core.search.service.mutator.mutator import Mutator


class ShuffleMutator(Mutator):

    """Shuffle mutator - block shuffling with Gaussian shift."""

    def mutate(self, individual: Individual) -> Individual:
        """Apply Gaussian shift to block center and reshuffle."""

        if len(individual.get_actions()) == 0:
            return self.add_shuffle_mutation()

        # Find block center
        actions = individual.get_actions()
        locations = [a.get_location() for a in actions]
        # Actions are stored as (row, col) due to Archive[x, y] access
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

        # Boundary checks (block: row-1 to row+1, col-1 to col+1)
        max_h = self.config["image_height"]
        max_w = self.config["image_width"]
        row = max(1, min(row, max_h - 2))
        col = max(1, min(col, max_w - 2))

        # Extract and shuffle block
        seed = self.archive.image.array.copy()
        sub_seed = seed[row - 1:row + 2, col - 1:col + 2, :]

        # Check if float image (0-1 range)
        is_float = False
        if sub_seed.dtype.kind == 'f' and np.max(sub_seed) <= 1.0:
            is_float = True

        rnd = np.reshape(sub_seed, (-1, 3))
        self.randomness.random.shuffle(rnd)
        sub_seed = np.reshape(rnd, (3, 3, 3))

        # Create new individual with shuffled block
        # Create new instance to effectively remove old actions for this block
        new_individual = Individual()
        for dr in range(-1, 2):
            for dc in range(-1, 2):

                # Get color values
                try:
                    r = sub_seed[dr + 1, dc + 1, 0]
                    g = sub_seed[dr + 1, dc + 1, 1]
                    b = sub_seed[dr + 1, dc + 1, 2]
                except IndexError:
                    continue

                # Convert to 0-255 int for Action
                if is_float:
                    r = int(r * 255)
                    g = int(g * 255)
                    b = int(b * 255)
                else:
                    r = int(r)
                    g = int(g)
                    b = int(b)

                new_individual.add_action(
                    Action(
                        location=(row + dr, col + dc),
                        red=r,
                        green=g,
                        blue=b,
                    )
                )

        return new_individual

    def add_shuffle_mutation(self) -> Individual:
        """Create initial shuffled block"""

        individual = Individual()
        seed = self.archive.image.array.copy()
        max_h = self.config["image_height"]
        max_w = self.config["image_width"]

        # Conflict Check Addition
        collision = True
        attempts = 0
        row, col = 0, 0

        # Get existing actions from archive for collision check
        archive_actions = []
        if self.archive and not self.archive.is_empty():
            archive_actions = self.archive.get_actions()

        while collision and attempts < 100:
            collision = False
            attempts += 1

            # Random position with even alignment
            row = self.randomness.next_int(2, max_h - 3)
            col = self.randomness.next_int(2, max_w - 3)

            if row % 2 == 1:
                row += 1
            if col % 2 == 1:
                col += 1

            # Check collision with existing block centers
            for action in archive_actions:
                a_row, a_col = action.get_location()
                if a_row == row and a_col == col:
                    collision = True
                    break

        # Shuffle Logic

        sub_seed = seed[row - 1:row + 2, col - 1:col + 2, :]

        # Check if float image (0-1 range)
        is_float = False
        if sub_seed.dtype.kind == 'f' and np.max(sub_seed) <= 1.0:
            is_float = True

        rnd = np.reshape(sub_seed, (-1, 3))
        self.randomness.random.shuffle(rnd)
        sub_seed = np.reshape(rnd, (3, 3, 3))

        for dr in range(-1, 2):
            for dc in range(-1, 2):

                # Get color values
                try:
                    r = sub_seed[dr + 1, dc + 1, 0]
                    g = sub_seed[dr + 1, dc + 1, 1]
                    b = sub_seed[dr + 1, dc + 1, 2]
                except IndexError:
                    continue

                # Convert to 0-255 int for Action
                if is_float:
                    r = int(r * 255)
                    g = int(g * 255)
                    b = int(b * 255)
                else:
                    r = int(r)
                    g = int(g)
                    b = int(b)

                individual.add_action(
                    Action(
                        location=(row + dr, col + dc),
                        red=r,
                        green=g,
                        blue=b,
                    )
                )

        return individual
