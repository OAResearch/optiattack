"""Shared base of the algorithms that run an attack on the NUT through the query tracker (baselines, white-box)."""
import logging
from typing import Any, Callable, List, Optional

import numpy as np

from core.config_parser import ConfigParser
from core.search.action import Action
from core.search.algorithms.baseline.query_tracker import STALL_LIMIT, AttackStalled, QueryTracker, StopAttack
from core.search.algorithms.search_algorithm import SearchAlgorithm
from core.search.evaluated_individual import EvaluatedIndividual
from core.search.fitness_value import FitnessValue
from core.search.individual import Individual

PRUNING_WARNING_ACTIONS = 1000
INSTALL_HINT = ("The '%s' algorithm needs the Adversarial Robustness Toolbox (ART). "
                "Install it with: pip install -r requirements-baselines.txt")


class AttackRunnerAlgorithm(SearchAlgorithm):

    """Runs an attack whose NUT queries go through a QueryTracker, and reports the tracker's best image."""

    tracker: Optional[QueryTracker] = None

    @property
    def logger(self) -> logging.Logger:
        """Logger of the concrete algorithm's module."""
        return logging.getLogger(type(self).__module__)

    def _import_art(self):
        """Imports ART and the remote estimator lazily; raises a RuntimeError with an install hint if ART is missing."""
        try:
            import art  # type: ignore[import-untyped]
            from core.search.algorithms.baseline.estimator import RemoteNutEstimator  # noqa: F401
        except ImportError as error:
            raise RuntimeError(INSTALL_HINT % self.get_type()) from error
        return art

    def _reject_max_pixels(self, description: str, alternatives: str) -> None:
        """Raises a ValueError (before any NUT query) when a pixel budget is set for an attack changing every pixel."""
        max_pixels = int(self.config.get("max_pixels", 0) or 0)
        if max_pixels > 0:
            raise ValueError("max_pixels=%d is not supported by the %s: it changes the whole image and cannot keep a "
                             "pixel budget. Use max_pixels 0, or %s." % (max_pixels, description, alternatives))

    def _is_targeted(self) -> bool:
        """True for a targeted attack."""
        return self.config.get("attack_type") == ConfigParser.AttackType.TARGETED

    def _label_index(self, labels: List[str], targeted: bool) -> int:
        """Index of the target class (targeted) or of the original class (untargeted) in the NUT class list."""
        if targeted:
            label = self.config.get("target")
            kind = "Target"
        else:
            label = self.archive.get_original_prediction_results().max_score.label
            kind = "Original"
        if label not in labels:
            raise ValueError("%s class '%s' is not in the NUT class list %s." % (kind, label, labels))
        return labels.index(label)

    @staticmethod
    def _one_hot(labels: List[str], index: int) -> np.ndarray:
        """One-hot label array of shape (1, classes) as ART expects it."""
        y = np.zeros((1, len(labels)), dtype=np.float32)
        y[0, index] = 1.0
        return y

    def _start_tracker(self, norm: float, stop_at_success: bool, reserved_queries: int = 0) -> QueryTracker:
        """Creates the query tracker with the original image as the best so far and makes its fitness current."""
        original = self.archive.image.array
        self.tracker = QueryTracker(self.stc, self.ff.fitness_from_result, original, norm, stop_at_success,
                                    self._on_new_best, reserved_queries=reserved_queries)
        # Only images that improve on the original become the best; without any, the original is reported.
        self.tracker.start_from_original(self.archive.get_original_prediction_results())
        self.stc.set_current_fitness(self.tracker.best_fitness)
        return self.tracker

    def _build_estimator(self, labels: List[str], label_only: bool = False, use_logits: bool = False,
                         allow_gradients: bool = False):
        """Creates the ART estimator on the NUT (after `_start_tracker`); only white-box use sets allow_gradients."""
        from core.search.algorithms.baseline.estimator import RemoteNutEstimator
        if self.tracker is None:
            raise RuntimeError("_start_tracker must be called before _build_estimator.")
        original = self.archive.image.array
        estimator = RemoteNutEstimator(self.ff.remote_controller, labels, tuple(original.shape), self.tracker,
                                       label_only=label_only, use_logits=use_logits, allow_gradients=allow_gradients)
        # The original was already scored by run_nut, which OptiAttack's own algorithms do not count either.
        estimator.seed_cache(original, self.archive.get_original_prediction_results())
        return estimator

    def _run_attack(self, run: Callable[[], Any], name: str, stall_hint: str = "",
                    queried: str = "predictions") -> Any:
        """Runs the attack and returns its result, or None when the tracker stopped it (budget, success or stall)."""
        try:
            return run()
        except AttackStalled:
            self._warn_about_stall(name, stall_hint, queried)
        except StopAttack:
            pass
        return None

    def _warn_about_stall(self, name: str, hint: str = "", queried: str = "predictions") -> None:
        """Warns that the attack was stopped because it only asked for results that were already known."""
        self.logger.warning("'%s' was stopped: it produced no new image in %d %s in a row, so the rest of the "
                            "budget could not be spent.%s", name, STALL_LIMIT, queried, hint)

    def _pruning_warning_actions(self) -> int:
        """Number of solution actions above which pruning is considered costly."""
        return PRUNING_WARNING_ACTIONS

    def _warn_about_pruning_cost(self) -> None:
        """Warns when pruning is enabled and the solution has many actions, since pruning queries the NUT a lot."""
        n_actions = len(self.archive.get_actions())
        if self.config.get("enable_pruning") is True and n_actions > self._pruning_warning_actions():
            self.logger.warning("The %s solution changes %d pixels; pruning it may need many NUT queries.",
                                self.get_type(), n_actions)

    def _finish(self):
        """Warns about the pruning cost and returns the archive's solution (the tracker's best image)."""
        self._warn_about_pruning_cost()
        return self.archive.extract_solution()

    def _on_new_best(self, image: np.ndarray, fitness: FitnessValue) -> None:
        """Makes the new best current at once; its actions are only built when the archive is read (can be costly)."""
        self.archive.set_pending(lambda: self._individual_from_best(image, fitness), fitness)

    def _individual_from_best(self, image: np.ndarray, fitness: FitnessValue) -> EvaluatedIndividual:
        """Individual with one action per pixel where the best image differs from the original."""
        original = self.archive.image.array
        rows, cols = np.nonzero(np.any(image != original, axis=-1))
        individual = Individual()
        individual.actions = [Action((int(r), int(c)), *(int(v) for v in image[r, c])) for r, c in zip(rows, cols)]
        return EvaluatedIndividual(individual, fitness)
