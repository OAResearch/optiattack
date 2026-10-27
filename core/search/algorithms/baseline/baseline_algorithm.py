"""Search algorithm that runs an ART baseline attack against the NUT under OptiAttack's budget and fitness."""
import logging
import random

import numpy as np

from core.config_parser import ConfigParser
from core.search.algorithms.baseline.attack_runner import (INSTALL_HINT as _INSTALL_HINT, PRUNING_WARNING_ACTIONS,
                                                           AttackRunnerAlgorithm)
from core.search.algorithms.baseline.attacks import ATTACKS, AttackSpec, resolve_stop_at_success
from core.search.algorithms.baseline.hsj_init import find_hsj_initial_image
from core.search.algorithms.baseline.nut_interface import NutRequirement, check_requirement

logger = logging.getLogger(__name__)

INSTALL_HINT = _INSTALL_HINT % ConfigParser.Algorithms.BASELINE


class BaselineAlgorithm(AttackRunnerAlgorithm):

    """Runs a baseline attack (ART's Square, HopSkipJump, SimBA or PixelAttack, or Sparse-RS) for OptiAttack."""

    def get_type(self):
        """Return the type of the search algorithm."""
        return ConfigParser.Algorithms.BASELINE

    def search(self):
        """Runs the selected ART attack until success or until the budget is used up, then returns the solution."""
        self.stc.start_search()
        spec = self._attack_spec()
        self._check_max_pixels(spec)
        art = self._import_art()

        logger.info("Running baseline '%s' with ART %s (report this version in papers).", spec.name, art.__version__)
        labels = check_requirement(spec.requirement, self.ff.remote_controller.get_nut_info())
        targeted = self._is_targeted()
        label_index = self._label_index(labels, targeted)
        self._warn_about_threshold(spec)
        stop_at_success = resolve_stop_at_success(spec, self.config)
        self._warn_about_stop_policy(spec, stop_at_success)

        original = self.archive.image.array
        self._start_tracker(spec.norm(self.config), stop_at_success)
        estimator = self._build_estimator(labels, label_only=spec.requirement == NutRequirement.LABEL)
        attack = spec.build(estimator, self.config, targeted)

        # ART draws from the global np.random (seeded by Randomness); some code paths use Python's `random`.
        random.seed(int(np.random.randint(0, 2 ** 31 - 1)))
        x = original[np.newaxis].astype(np.float32)
        y = self._one_hot(labels, label_index)

        def run() -> None:
            """Finds HopSkipJump's start image if needed, then runs the attack."""
            kwargs = {}
            if spec.name == "hsj":
                init = find_hsj_initial_image(estimator, original, self.randomness, self.config,
                                              lambda vector: (int(np.argmax(vector)) == label_index) == targeted)
                if init is None:
                    logger.warning("No adversarial start image was found for hsj within hsj_init_size=%s tries; "
                                   "HopSkipJump is not run. Increase hsj_init_size or give --hsj_init_image.",
                                   self.config.get("hsj_init_size"))
                    return
                kwargs["x_adv_init"] = init[np.newaxis]
            attack.generate(x=x, y=y, **kwargs)

        self._run_attack(run, spec.name, self._stall_hint(spec))
        return self._finish()

    def _attack_spec(self) -> AttackSpec:
        """Returns the definition of the attack selected with `baseline_attack`."""
        name = str(self.config.get("baseline_attack")).strip().lower()
        if name not in ATTACKS:
            raise ValueError("Unknown baseline_attack '%s'. Options: %s." % (name, sorted(ATTACKS)))
        return ATTACKS[name]

    def _check_max_pixels(self, spec: AttackSpec) -> None:
        """Rejects max_pixels for the dense attacks, and a pixel_th or sparse_rs_k above it, before any NUT query."""
        max_pixels = int(self.config.get("max_pixels", 0) or 0)
        if max_pixels <= 0:
            return
        if spec.name in ("square", "simba", "hsj"):
            self._reject_max_pixels("dense baseline '%s'" % spec.name,
                                    "an L0 baseline: 'sparse_rs', or 'pixel' with pixel_th <= max_pixels")
        if spec.name == "pixel":
            pixel_th = int(self.config.get("pixel_th", 0) or 0)
            if pixel_th > max_pixels:
                raise ValueError("pixel_th=%d is above max_pixels=%d: PixelAttack would change more pixels than the "
                                 "budget allows. Use pixel_th <= max_pixels." % (pixel_th, max_pixels))
        if spec.name == "sparse_rs":
            sparse_rs_k = int(self.config.get("sparse_rs_k", 0) or 0)
            if sparse_rs_k > max_pixels:
                raise ValueError("sparse_rs_k=%d is above max_pixels=%d: Sparse-RS would change more pixels than "
                                 "the budget allows. Use sparse_rs_k <= max_pixels, or sparse_rs_k 0 to use "
                                 "max_pixels." % (sparse_rs_k, max_pixels))

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

    def _stall_hint(self, spec: AttackSpec) -> str:
        """Advice appended to the stall warning (SimBA's 'dct' steps vanish when rounded to uint8)."""
        if spec.name == "simba" and self.config.get("simba_attack") == "dct":
            return (" SimBA's 'dct' steps are spread over the whole image and vanish when rounded to uint8; use "
                    "--simba_attack px or a larger simba_epsilon.")
        return ""

    def _pruning_warning_actions(self) -> int:
        """Number of solution actions above which pruning is considered costly (this module's constant)."""
        return PRUNING_WARNING_ACTIONS
