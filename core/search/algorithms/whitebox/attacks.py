"""Definitions of the white-box attacks (ART's FGSM, PGD and C&W L2) run through the NUT's gradients."""
from typing import Dict

import numpy as np

from core.search.algorithms.baseline.attacks import AttackSpec, parse_norm
from core.search.algorithms.baseline.nut_interface import NutRequirement


def _build_fgsm(estimator, config: dict, targeted: bool):
    """Builds ART's FastGradientMethod in L-inf (FGSM): one loss gradient."""
    from art.attacks.evasion import FastGradientMethod  # type: ignore[import-untyped]
    return FastGradientMethod(estimator, norm=np.inf, eps=float(config["fgsm_eps"]), targeted=targeted,
                              num_random_init=0, batch_size=1)


def _build_pgd(estimator, config: dict, targeted: bool):
    """Builds ART's ProjectedGradientDescent (numpy version): one loss gradient per iteration."""
    from art.attacks.evasion import ProjectedGradientDescent  # type: ignore[import-untyped]
    return ProjectedGradientDescent(estimator, norm=parse_norm(config, "pgd_norm"), eps=float(config["pgd_eps"]),
                                    eps_step=float(config["pgd_eps_step"]), max_iter=int(config["pgd_max_iter"]),
                                    targeted=targeted, num_random_init=0, batch_size=1, verbose=False)


def _build_cw(estimator, config: dict, targeted: bool):
    """Builds ART's CarliniL2Method: two class (logit) gradients and a line search of predictions per iteration."""
    from art.attacks.evasion import CarliniL2Method  # type: ignore[import-untyped]
    return CarliniL2Method(estimator, confidence=float(config["cw_confidence"]), targeted=targeted,
                           learning_rate=float(config["cw_learning_rate"]),
                           binary_search_steps=int(config["cw_binary_search_steps"]),
                           max_iter=int(config["cw_max_iter"]), initial_const=float(config["cw_initial_const"]),
                           max_halving=int(config["cw_max_halving"]), max_doubling=int(config["cw_max_doubling"]),
                           batch_size=1, verbose=False)


WHITEBOX_ATTACKS: Dict[str, AttackSpec] = {
    "fgsm": AttackSpec("fgsm", NutRequirement.GRADIENTS, False, lambda config: np.inf, _build_fgsm,
                       lambda config: 1),
    "pgd": AttackSpec("pgd", NutRequirement.GRADIENTS, False, lambda config: parse_norm(config, "pgd_norm"),
                      _build_pgd, lambda config: int(config["pgd_max_iter"])),
    "cw": AttackSpec("cw", NutRequirement.GRADIENTS, False, lambda config: 2, _build_cw,
                     lambda config: 2 * int(config["cw_binary_search_steps"]) * int(config["cw_max_iter"])),
}
