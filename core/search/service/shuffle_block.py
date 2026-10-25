"""Shared block logic of the shuffle sampler and the shuffle mutator."""
from typing import Optional

import numpy as np
from core.search.action import Action
from core.search.individual import Individual
from core.search.service.archive import Archive
from core.search.service.randomness import Randomness


class ShuffleBlockMixin:

    """Block helpers for classes that provide `config` and `randomness`."""

    # Provided by the sampler or mutator this mixin is combined with
    config: dict
    randomness: Randomness

    def _get_block_size(self) -> int:
        """Get block size from config, default is 3."""
        return self.config.get("shuffle_block_size", 3)

    def _get_half_size(self) -> int:
        """Get half block size for offset calculations."""
        return self._get_block_size() // 2

    def _is_odd_block(self) -> bool:
        """Check if block size is odd."""
        return self._get_block_size() % 2 == 1

    def _clamp_position(self, row: int, col: int) -> tuple[int, int]:
        """Clamp block center position to valid image bounds."""
        half = self._get_half_size()
        max_h = self.config["image_height"]
        max_w = self.config["image_width"]

        if self._is_odd_block():
            # For odd: block goes from row-half to row+half (inclusive)
            row = max(half, min(row, max_h - half - 1))
            col = max(half, min(col, max_w - half - 1))
        else:
            # For even: block goes from row-half to row+half-1 (inclusive)
            # Need room for 'half' pixels above and 'half' pixels below (including center-ish)
            row = max(half, min(row, max_h - half))
            col = max(half, min(col, max_w - half))

        return row, col

    def _extract_block(self, seed: np.ndarray, row: int, col: int) -> np.ndarray:
        """Extract a block from the image centered at (row, col)."""
        half = self._get_half_size()

        if self._is_odd_block():
            # Odd: [row-half : row+half+1] gives exactly block_size elements
            return seed[row - half:row + half + 1, col - half:col + half + 1, :]
        else:
            # Even: [row-half : row+half] gives exactly block_size elements
            return seed[row - half:row + half, col - half:col + half, :]

    def _shuffle_block(self, block: np.ndarray) -> np.ndarray:
        """Shuffle pixels within a block."""
        block_size = self._get_block_size()
        # Flatten to (N*N, 3), shuffle, reshape back
        flat = np.reshape(block, (-1, 3))
        self.randomness.random.shuffle(flat)
        return np.reshape(flat, (block_size, block_size, 3))

    def _is_float_image(self, block: np.ndarray) -> bool:
        """Check if image uses float (0-1) range."""
        return block.dtype.kind == 'f' and np.max(block) <= 1.0

    def _get_offset_range(self) -> range:
        """Get the offset range for iterating over block pixels."""

        half = self._get_half_size()
        if self._is_odd_block():
            return range(-half, half + 1)
        else:
            return range(-half, half)

    def _create_actions_from_block(self, individual: Individual, block: np.ndarray,
                                   row: int, col: int, is_float: bool) -> None:
        """Create actions from a block and add them to an individual."""
        offset_range = self._get_offset_range()

        for i, dr in enumerate(offset_range):
            for j, dc in enumerate(offset_range):
                try:
                    r = block[i, j, 0]
                    g = block[i, j, 1]
                    b = block[i, j, 2]
                except IndexError:
                    continue

                # Convert to 0-255 int for Action
                if is_float:
                    r, g, b = int(r * 255), int(g * 255), int(b * 255)
                else:
                    r, g, b = int(r), int(g), int(b)

                individual.add_action(
                    Action(
                        location=(row + dr, col + dc),
                        red=r,
                        green=g,
                        blue=b,
                    )
                )

    def sample_shuffled_block(self, archive: Optional[Archive]) -> Individual:
        """Create a shuffled block at a random position that does not collide with archived block centers."""
        individual = Individual()

        if archive is None:
            raise ValueError("Archive is required for shuffle mutation")

        seed = archive.image.array.copy()
        max_h = self.config["image_height"]
        max_w = self.config["image_width"]
        half = self._get_half_size()
        block_size = self._get_block_size()

        # Find non-colliding position
        collision = True
        attempts = 0
        row, col = 0, 0

        archive_actions = []
        if not archive.is_empty():
            archive_actions = archive.get_actions()

        while collision and attempts < 100:
            collision = False
            attempts += 1

            # Random position ensuring block fits within image
            if self._is_odd_block():
                row = self.randomness.next_int(half + 1, max_h - half - 2)
                col = self.randomness.next_int(half + 1, max_w - half - 2)
            else:
                row = self.randomness.next_int(half + 1, max_h - half - 1)
                col = self.randomness.next_int(half + 1, max_w - half - 1)

            # Align to block_size grid for even distribution
            row = (row // block_size) * block_size + half
            col = (col // block_size) * block_size + half

            # Check collision with existing block centers
            for action in archive_actions:
                a_row, a_col = action.get_location()
                if a_row == row and a_col == col:
                    collision = True
                    break

        # Clamp to valid bounds
        row, col = self._clamp_position(row, col)

        # Extract, shuffle, and create actions
        block = self._extract_block(seed, row, col)
        is_float = self._is_float_image(block)
        shuffled_block = self._shuffle_block(block)

        self._create_actions_from_block(individual, shuffled_block, row, col, is_float)

        return individual
