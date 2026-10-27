"""Search algorithm that runs an ART white-box attack (FGSM, PGD, C&W L2) with the NUT's gradients."""
import logging
from typing import Optional

import numpy as np

from core.config_parser import ConfigParser
from core.search.algorithms.baseline.attack_runner import AttackRunnerAlgorithm
from core.search.algorithms.baseline.attacks import AttackSpec
from core.search.algorithms.baseline.nut_interface import check_requirement, to_logit_vector
from core.search.algorithms.baseline.query_tracker import StopAttack
from core.search.algorithms.whitebox.attacks import WHITEBOX_ATTACKS

logger = logging.getLogger(__name__)

# Parameters that set how many gradient queries an attack makes, named in the budget warning.
_ITERATION_PARAMETERS = {"fgsm": "", "pgd": "pgd_max_iter", "cw": "cw_binary_search_steps and cw_max_iter"}


class WhiteboxAlgorithm(AttackRunnerAlgorithm):

    """
    Runs ART's FGSM, PGD or C&W L2 on a NUT that provides gradients and logits ('whitebox' algorithm).

    Gradient and prediction queries share `max_evaluations` (a gradient query counts as one query). The attack may
    use all but one query; the last one is kept to predict the attack's result, rounded to uint8, with a normal
    prediction query. If the budget stops the attack first, the image it was last working on is checked instead.
    Success and the reported image follow the query tracker, as for the black-box baselines.
    """

    attack_completed: Optional[bool] = None

    def get_type(self):
        """Return the type of the search algorithm."""
        return ConfigParser.Algorithms.WHITEBOX

    def search(self):
        """Checks the NUT's gradients and logits, runs the attack, verifies its uint8 result, returns the solution."""
        self.stc.start_search()
        spec = self._attack_spec()
        self._reject_max_pixels("white-box attack '%s'" % spec.name,
                                "an L0 attack: OptiAttack's algorithms, or the baselines 'sparse_rs' or 'pixel'")
        art = self._import_art()

        logger.info("Running white-box attack '%s' with ART %s (report this version in papers).", spec.name,
                    art.__version__)
        labels = check_requirement(spec.requirement, self.ff.remote_controller.get_nut_info())
        # The original result comes from run_nut: a NUT without logits fails here, before any query.
        to_logit_vector(self.archive.get_original_prediction_results(), labels)
        targeted = self._is_targeted()
        label_index = self._label_index(labels, targeted)

        self._start_tracker(spec.norm(self.config), stop_at_success=False, reserved_queries=1)
        estimator = self._build_estimator(labels, use_logits=True, allow_gradients=True)
        attack = spec.build(estimator, self.config, targeted)
        self._warn_about_budget(spec)

        x = self.archive.image.array[np.newaxis].astype(np.float32)
        y = self._one_hot(labels, label_index)
        output = self._run_attack(lambda: attack.generate(x=x, y=y), spec.name,
                                  queried="predictions or gradients")
        self.attack_completed = output is not None
        if output is None:
            logger.info("'%s' was stopped before it finished; its last image is checked instead.", spec.name)
        self._verify_final_image(estimator, None if output is None else np.asarray(output)[0])
        return self._finish()

    def _attack_spec(self) -> AttackSpec:
        """Returns the definition of the attack selected with `whitebox_attack`."""
        name = str(self.config.get("whitebox_attack")).strip().lower()
        if name not in WHITEBOX_ATTACKS:
            raise ValueError("Unknown whitebox_attack '%s'. Options: %s." % (name, sorted(WHITEBOX_ATTACKS)))
        return WHITEBOX_ATTACKS[name]

    def _verify_final_image(self, estimator, output: Optional[np.ndarray]) -> None:
        """Predicts the attack's result (or its last image) as uint8 with the reserved query, through the tracker."""
        image = output if output is not None else estimator.last_input
        if image is None or self.tracker is None:
            return
        self.tracker.reserved_queries = 0
        try:
            # Rounded to uint8 by the estimator; a cache hit if the attack already predicted this image.
            estimator.predict(np.asarray(image, dtype=np.float32)[np.newaxis])
        except StopAttack:
            pass

    def _warn_about_budget(self, spec: AttackSpec) -> None:
        """Warns when the configured iterations need clearly more gradient queries than the budget leaves."""
        if self.config.get("stopping_criterion") != ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS:
            return
        needed = spec.gradient_calls(self.config)
        available = int(self.config["max_evaluations"]) - self.stc.get_evaluated_individuals() - 1
        if needed is None or needed <= available:
            return
        logger.warning("'%s' needs at least %d gradient queries with these settings, but only %d of max_evaluations "
                       "are left for it (one is kept for the final check). It will be stopped by the budget and its "
                       "best image so far is reported; lower %s or raise max_evaluations.", spec.name, needed,
                       max(available, 0), _ITERATION_PARAMETERS.get(spec.name) or "the iterations")
