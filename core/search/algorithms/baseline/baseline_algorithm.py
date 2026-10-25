"""Search algorithm that runs an ART baseline attack against the NUT under OptiAttack's budget and fitness."""
import logging
import random
from typing import List, Optional

import numpy as np

from core.config_parser import ConfigParser
from core.search.action import Action
from core.search.algorithms.baseline.attacks import ATTACKS, AttackSpec, resolve_stop_at_success
from core.search.algorithms.baseline.hsj_init import find_hsj_initial_image
from core.search.algorithms.baseline.nut_interface import NutRequirement, check_requirement
from core.search.algorithms.baseline.query_tracker import STALL_LIMIT, AttackStalled, QueryTracker, StopAttack
from core.search.algorithms.search_algorithm import SearchAlgorithm
from core.search.evaluated_individual import EvaluatedIndividual
from core.search.fitness_value import FitnessValue
from core.search.individual import Individual

logger = logging.getLogger(__name__)

PRUNING_WARNING_ACTIONS = 1000
INSTALL_HINT = ("The 'baseline' algorithm needs the Adversarial Robustness Toolbox (ART). "
                "Install it with: pip install -r requirements-baselines.txt")


class BaselineAlgorithm(SearchAlgorithm):

    """Runs an ART attack (Square, HopSkipJump, SimBA or PixelAttack) as a baseline for OptiAttack."""

    tracker: Optional[QueryTracker] = None

    def get_type(self):
        """Return the type of the search algorithm."""
        return ConfigParser.Algorithms.BASELINE

    def search(self):
        """Runs the selected ART attack until success or until the budget is used up, then returns the solution."""
        self.stc.start_search()
        try:
            import art  # type: ignore[import-untyped]
            from core.search.algorithms.baseline.estimator import RemoteNutEstimator
        except ImportError as error:
            raise RuntimeError(INSTALL_HINT) from error

        spec = self._attack_spec()
        logger.info("Running baseline '%s' with ART %s (report this version in papers).", spec.name, art.__version__)
        remote_controller = self.ff.remote_controller
        labels = check_requirement(spec.requirement, remote_controller.get_nut_info())
        targeted = self.config.get("attack_type") == ConfigParser.AttackType.TARGETED
        label_index = self._label_index(labels, targeted)
        self._warn_about_threshold(spec)
        stop_at_success = resolve_stop_at_success(spec, self.config)
        self._warn_about_stop_policy(spec, stop_at_success)

        original = self.archive.image.array
        original_result = self.archive.get_original_prediction_results()
        self.tracker = QueryTracker(self.stc, self.ff.fitness_from_result, original, spec.norm(self.config),
                                    stop_at_success, self._on_new_best)
        # Only images that improve on the original become the best; without any, the original is reported.
        self.tracker.start_from_original(original_result)
        self.stc.set_current_fitness(self.tracker.best_fitness)
        estimator = RemoteNutEstimator(remote_controller, labels, tuple(original.shape), self.tracker,
                                       label_only=spec.requirement == NutRequirement.LABEL)
        # The original was already scored by run_nut, which OptiAttack's own algorithms do not count either.
        estimator.seed_cache(original, original_result)
        attack = spec.build(estimator, self.config, targeted)

        # ART draws from the global np.random (seeded by Randomness); some code paths use Python's `random`.
        random.seed(int(np.random.randint(0, 2 ** 31 - 1)))
        x = original[np.newaxis].astype(np.float32)
        y = np.zeros((1, len(labels)), dtype=np.float32)
        y[0, label_index] = 1.0
        try:
            kwargs = {}
            run = True
            if spec.name == "hsj":
                init = find_hsj_initial_image(estimator, original, self.randomness, self.config,
                                              lambda vector: (int(np.argmax(vector)) == label_index) == targeted)
                if init is None:
                    logger.warning("No adversarial start image was found for hsj within hsj_init_size=%s tries; "
                                   "HopSkipJump is not run. Increase hsj_init_size or give --hsj_init_image.",
                                   self.config.get("hsj_init_size"))
                    run = False
                else:
                    kwargs["x_adv_init"] = init[np.newaxis]
            if run:
                attack.generate(x=x, y=y, **kwargs)
        except AttackStalled:
            self._warn_about_stall(spec)
        except StopAttack:
            pass

        self._warn_about_pruning_cost()
        return self.archive.extract_solution()

    def _attack_spec(self) -> AttackSpec:
        """Returns the definition of the attack selected with `baseline_attack`."""
        name = str(self.config.get("baseline_attack")).strip().lower()
        if name not in ATTACKS:
            raise ValueError("Unknown baseline_attack '%s'. Options: %s." % (name, sorted(ATTACKS)))
        return ATTACKS[name]

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

    def _warn_about_threshold(self, spec: AttackSpec) -> None:
        """Warns that SimBA, PixelAttack and HopSkipJump work on the class change and ignore a non-zero threshold."""
        threshold = self.config.get("fitness_threshold", 0.0)
        if threshold == 0:
            return
        if spec.name in ("simba", "pixel"):
            logger.warning("fitness_threshold=%s is not fully supported by '%s': the attack stops by itself once "
                           "the class changes, so the threshold may not be reached.", threshold, spec.name)
        elif spec.name == "hsj":
            logger.warning("fitness_threshold=%s is not fully supported by 'hsj': HopSkipJump only sees labels and "
                           "stays near the decision boundary, so a negative threshold is practically unreachable.",
                           threshold)

    @staticmethod
    def _warn_about_stop_policy(spec: AttackSpec, stop_at_success: bool) -> None:
        """Warns that baseline_stop_at_success=false cannot keep SimBA and PixelAttack running after a success."""
        if spec.name in ("simba", "pixel") and not stop_at_success:
            logger.warning("baseline_stop_at_success=false has no effect for '%s': ART's attack stops by itself "
                           "once the class changes.", spec.name)

    def _warn_about_stall(self, spec: AttackSpec) -> None:
        """Warns that the attack was stopped because it only produced images that were already scored."""
        hint = ""
        if spec.name == "simba" and self.config.get("simba_attack") == "dct":
            hint = (" SimBA's 'dct' steps are spread over the whole image and vanish when rounded to uint8; use "
                    "--simba_attack px or a larger simba_epsilon.")
        logger.warning("'%s' was stopped: it produced no new image in %d predictions in a row, so the rest of the "
                       "budget could not be spent.%s", spec.name, STALL_LIMIT, hint)

    def _warn_about_pruning_cost(self) -> None:
        """Warns when pruning is enabled and the solution has many actions, since pruning queries the NUT a lot."""
        n_actions = len(self.archive.get_actions())
        if self.config.get("enable_pruning") is True and n_actions > PRUNING_WARNING_ACTIONS:
            logger.warning("The baseline solution changes %d pixels; pruning it may need many NUT queries.",
                           n_actions)

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
