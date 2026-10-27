"""Pixel budget (--max_pixels): repairs candidates so that no image sent to the NUT changes more than k pixels."""
from typing import List, Set, Tuple

import numpy as np

from core.search.action import Action
from core.search.individual import Individual
from core.search.service.archive import Archive
from core.search.service.randomness import Randomness


class PixelBudget:

    """Caps the number of changed pixels (archive actions plus candidate actions) by repairing the candidate."""

    def __init__(self, config: dict, randomness: Randomness, archive: Archive) -> None:
        """Initialize the pixel budget from the `max_pixels` parameter (0: no limit)."""
        limit = int(config.get("max_pixels", 0) or 0)
        if limit < 0:
            raise ValueError("max_pixels must be 0 (no limit) or a positive number of pixels, got %d." % limit)
        self._limit = limit
        self.randomness = randomness
        self.archive = archive

    @property
    def limit(self) -> int:
        """Maximum number of changed pixels; 0 means unlimited."""
        return self._limit

    def repair(self, individual: Individual) -> bool:
        """Move excess new-pixel actions (same colour) onto random changed pixels so the image fits; False if empty."""
        if self._limit <= 0 or not individual.actions:
            return len(individual.actions) > 0

        original = self.archive.image.array
        base = self.archive.get_mutated_image()
        archive_changed = np.any(base != original, axis=-1)
        used = int(np.count_nonzero(archive_changed))
        if used > self._limit:
            raise RuntimeError("The archive already changes %d pixels, more than max_pixels=%d; the pixel budget "
                               "cannot be kept." % (used, self._limit))

        candidate = individual.get_action_image(base.copy())
        candidate_changed = np.any(candidate != original, axis=-1)
        # New locations: changed by the candidate, not by the archive; in order of first appearance (seeded choices).
        new_locations: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        for action in individual.actions:
            location = self._location(action)
            if location not in seen and candidate_changed[location] and not archive_changed[location]:
                new_locations.append(location)
            seen.add(location)

        allowed = max(self._limit - used, 0)
        excess: Set[Tuple[int, int]] = set()
        while len(new_locations) > allowed:
            excess.add(new_locations.pop(int(self.randomness.random_choice(len(new_locations)))))
        if not excess:
            return True

        # Already-changed pixels: the archive's (A) plus the new locations that were kept.
        targets = [(int(r), int(c)) for r, c in np.argwhere(archive_changed)] + new_locations
        repaired = []
        for action in individual.actions:
            if self._location(action) not in excess:
                repaired.append(action)
            elif targets:
                moved = action.copy()
                moved.set_location(*targets[int(self.randomness.random_choice(len(targets)))])
                repaired.append(moved)
            # else: no changed pixel exists at all (|A| == 0 and allowed == 0); the action is dropped.
        individual.actions[:] = repaired
        return len(individual.actions) > 0

    @staticmethod
    def _location(action: Action) -> Tuple[int, int]:
        """Location of an action as a tuple of Python ints."""
        row, col = action.get_location()
        return int(row), int(col)
