"""Tracks every NUT query of an ART attack: enforces the budget, computes fitness and keeps the best image."""
from typing import Callable, Optional

import numpy as np

from core.search.fitness_value import FitnessValue
from core.utils.nut_request import NutRequest

# Consecutive predictions answered from the cache after which an attack is considered stalled.
STALL_LIMIT = 1000


class StopAttack(Exception):

    """Base exception used to interrupt an ART attack."""


class BudgetExhausted(StopAttack):

    """Raised when the evaluation budget is used up."""


class SolutionFound(StopAttack):

    """Raised when a successful adversarial image is found and the attack should stop."""


class AttackStalled(StopAttack):

    """Raised when the attack keeps predicting known images, e.g. because its steps vanish when rounded to uint8."""


class QueryTracker:

    """Watches NUT queries made by an ART attack and tracks the best image found."""

    def __init__(self, stc, fitness_from_result: Callable[[NutRequest], float], original: np.ndarray,
                 norm: float, stop_at_success: bool, on_new_best: Callable[[np.ndarray, FitnessValue], None]):
        """Initializes the tracker with the budget controller, fitness function and the original image."""
        self.stc = stc
        self.fitness_from_result = fitness_from_result
        self.original = original
        self.norm = norm
        self.stop_at_success = stop_at_success
        self.on_new_best = on_new_best
        self.best_image: Optional[np.ndarray] = None
        self.best_fitness: Optional[FitnessValue] = None
        self._best_distance = float("inf")
        self._best_is_success = False
        self._cache_hits_in_a_row = 0

    def start_from_original(self, result: NutRequest) -> None:
        """Makes the original image the best so far, so only queries that improve on it become the best."""
        self.best_image = np.array(self.original, copy=True)
        self.best_fitness = FitnessValue(self.fitness_from_result(result), result.predictions)

    def cache_hit(self) -> None:
        """Counts a prediction answered without a NUT query; raises AttackStalled after STALL_LIMIT in a row."""
        self._cache_hits_in_a_row += 1
        if self._cache_hits_in_a_row >= STALL_LIMIT:
            raise AttackStalled()

    def before_query(self) -> None:
        """Raises BudgetExhausted if the budget is already used up."""
        if self.stc.percentage_used_budget() >= 1.0:
            raise BudgetExhausted()

    def distance(self, image: np.ndarray) -> float:
        """Perturbation size of the image w.r.t. the original in the attack's norm."""
        diff = np.asarray(image, dtype=np.float64) - np.asarray(self.original, dtype=np.float64)
        if self.norm == 0:
            return float(np.count_nonzero(np.any(diff != 0, axis=-1)))
        return float(np.linalg.norm(diff.ravel(), ord=self.norm))

    def after_query(self, image: np.ndarray, result: NutRequest) -> None:
        """Computes the fitness, updates the best image and stops the attack on success if requested."""
        self._cache_hits_in_a_row = 0
        fitness = FitnessValue(self.fitness_from_result(result), result.predictions)
        success = bool(self.stc.is_solution(fitness))
        dist = self.distance(image) if success else float("inf")

        if self._is_better(fitness, success, dist):
            self.best_image = np.array(image, copy=True)
            self.best_fitness = fitness
            self._best_distance = dist
            self._best_is_success = success
            self.on_new_best(self.best_image, fitness)

        if success and self.stop_at_success:
            raise SolutionFound()

    def _is_better(self, fitness: FitnessValue, success: bool, dist: float) -> bool:
        """Success beats failure; among successes the smaller perturbation wins, else lower fitness."""
        if self.best_fitness is None:
            return True
        if success:
            return not self._best_is_success or dist < self._best_distance
        return not self._best_is_success and fitness.value < self.best_fitness.value
