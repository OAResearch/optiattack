"""Shuffle sampler implementation - creates blocks samples for ShuffleMutator."""

import numpy as np
from core.search.action import Action
from core.search.individual import Individual
from core.search.service.sampler.sampler import Sampler


class ShuffleSampler(Sampler):

    """Shuffle sampler that creates block samples compatible with ShuffleMutator."""

    def sample(self):
        """Sample an individual with a shuffled block."""
        individual = Individual()
        seed = self.archive.image.array.copy()
        max_h = self.config["image_height"]
        max_w = self.config["image_width"]

        # Conflict check
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

        # Shuffle logic
        sub_seed = seed[row - 1:row + 2, col - 1:col + 2, :]

        # Check if float image (0-1 range)
        is_float = False
        if sub_seed.dtype.kind == 'f' and np.max(sub_seed) <= 1.0:
            is_float = True

        rnd = np.reshape(sub_seed, (-1, 3))
        self.randomness.random.shuffle(rnd)
        sub_seed = np.reshape(rnd, (3, 3, 3))

        # Create actions
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
