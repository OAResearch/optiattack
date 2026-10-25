"""Definitions of the ART baseline attacks: what they need from the NUT, their norm and how they are built."""
from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np

from core.config_parser import ConfigParser
from core.search.algorithms.baseline.nut_interface import NutRequirement

_NORMS = {"inf": np.inf, "2": 2}


def _parse_norm(config: dict, key: str) -> float:
    """Converts a norm parameter ('inf' or '2') to the value used by ART and the query tracker."""
    value = str(config.get(key)).strip().lower()
    if value not in _NORMS:
        raise ValueError("Invalid value '%s' for %s. Options: %s." % (config.get(key), key, sorted(_NORMS)))
    return _NORMS[value]


@dataclass(frozen=True)
class AttackSpec:

    """Describes one ART baseline attack."""

    name: str
    requirement: NutRequirement
    stop_at_success_default: bool
    norm_fn: Callable[[dict], float]
    build_fn: Callable[[Any, dict, bool], Any]

    def norm(self, config: dict) -> float:
        """Norm in which the attack measures the perturbation (0 means number of changed pixels)."""
        return self.norm_fn(config)

    def build(self, estimator, config: dict, targeted: bool):
        """Builds the ART attack (an `art.attacks.EvasionAttack`) for the given estimator."""
        return self.build_fn(estimator, config, targeted)


def resolve_stop_at_success(spec: AttackSpec, config: dict) -> bool:
    """Resolves `baseline_stop_at_success` ('auto', 'true' or 'false') for the given attack."""
    value = str(config.get("baseline_stop_at_success", "auto")).strip().lower()
    if value == "auto":
        return spec.stop_at_success_default
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("Invalid value '%s' for baseline_stop_at_success. Options: 'auto', 'true', 'false'."
                     % config.get("baseline_stop_at_success"))


def _never_adversarial(y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Square adversarial criterion that never stops the attack; the query tracker decides instead."""
    return np.zeros(len(y_pred), dtype=bool)


def _targeted_margin_loss(estimator) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Square loss for targeted mode: best non-target score minus target score (y is the target one-hot)."""
    def loss(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Margin of the target class to the best other class; negative once the target is on top."""
        y_pred = np.asarray(estimator.predict(x))
        rows = np.arange(len(y_pred))
        target = np.argmax(y, axis=1)
        target_scores = y_pred[rows, target]
        others = np.array(y_pred, dtype=float, copy=True)
        others[rows, target] = -np.inf
        return others.max(axis=1) - target_scores
    return loss


def _max_square_side(norm: float, shorter_side: int) -> int:
    """Largest square side ART can place strictly inside the image (L2 squares are odd and at least 3 wide)."""
    side = shorter_side - 1
    if norm == 2 and side % 2 == 0:
        side -= 1
    return side


def _square_attack_class():
    """
    Returns ART's SquareAttack with the square size capped below the shorter image side (ART workaround).

    ART draws the square position with `randint(0, side - square_size)`, which fails when the square is as large as
    the shorter side (small or strongly non-square images). On the L2 path ART also enlarges the square to an odd
    size of at least 3. Capping only changes the schedule where stock ART would crash.
    """
    from art.attacks.evasion import SquareAttack  # type: ignore[import-untyped]

    class CappedSquareAttack(SquareAttack):

        """SquareAttack whose squares always fit strictly inside the image."""

        def _get_percentage_of_elements(self, i_iter: int) -> float:
            """Square size schedule of ART, capped so that the square side stays below the shorter image side."""
            height, width = self.estimator.input_shape[0], self.estimator.input_shape[1]
            side = _max_square_side(self.norm, min(height, width))
            return min(super()._get_percentage_of_elements(i_iter), side ** 2 / float(height * width))

    return CappedSquareAttack


def _check_square_image_size(norm: float, height: int, width: int) -> None:
    """Rejects image sizes on which ART's SquareAttack would crash or produce NaNs (ART workaround)."""
    min_side = 5 if norm == 2 else 2
    if min(height, width) < min_side:
        raise ValueError("Square attack with this square_norm needs image sides of at least %d pixels (got %dx%d)."
                         % (min_side, height, width))
    if norm == 2 and width < 5 * (height // 5):
        # ART's L2 start lays 5x5 tiles of side height // 5 along both axes.
        raise ValueError("ART's Square L2 initialisation needs width >= 5 * (height // 5) (got %dx%d). "
                         "Use --square_norm inf for this image." % (height, width))


def _square_max_iter(config: dict) -> int:
    """Resolves `square_max_iter`: 0 means the evaluation budget (or 10000 for a time budget)."""
    max_iter = int(config["square_max_iter"])
    if max_iter > 0:
        return max_iter
    if config.get("stopping_criterion") == ConfigParser.StoppingCriterion.TIME:
        return 10000
    return int(config["max_evaluations"])


def _build_square(estimator, config: dict, targeted: bool):
    """Builds ART's SquareAttack; targeted mode uses a target margin loss since ART has no `targeted` flag."""
    norm = _parse_norm(config, "square_norm")
    _check_square_image_size(norm, estimator.input_shape[0], estimator.input_shape[1])
    return _square_attack_class()(
        estimator,
        norm=norm,
        adv_criterion=_never_adversarial,
        loss=_targeted_margin_loss(estimator) if targeted else None,
        max_iter=_square_max_iter(config),
        eps=float(config["square_eps"]),
        p_init=float(config["square_p_init"]),
        nb_restarts=int(config["square_nb_restarts"]),
        batch_size=1,
        verbose=False,
    )


def _build_hsj(estimator, config: dict, targeted: bool):
    """Builds ART's HopSkipJump."""
    from art.attacks.evasion import HopSkipJump  # type: ignore[import-untyped]
    return HopSkipJump(
        estimator,
        batch_size=1,
        targeted=targeted,
        norm=_parse_norm(config, "hsj_norm"),
        max_iter=int(config["hsj_max_iter"]),
        max_eval=int(config["hsj_max_eval"]),
        init_eval=int(config["hsj_init_eval"]),
        init_size=int(config["hsj_init_size"]),
        verbose=False,
    )


def _build_simba(estimator, config: dict, targeted: bool):
    """Builds ART's SimBA; rejects non-square images for 'dct'/'diag', which ART silently mishandles (workaround)."""
    from art.attacks.evasion import SimBA  # type: ignore[import-untyped]
    height, width = estimator.input_shape[0], estimator.input_shape[1]
    if height != width and (config["simba_attack"] == "dct" or config["simba_order"] == "diag"):
        raise ValueError("ART's SimBA supports simba_attack dct and simba_order diag only on square images "
                         "(got %dx%d). Use --simba_attack px --simba_order random." % (height, width))
    return SimBA(
        estimator,
        attack=str(config["simba_attack"]),
        max_iter=int(config["simba_max_iter"]),
        order=str(config["simba_order"]),
        epsilon=float(config["simba_epsilon"]),
        freq_dim=int(config["simba_freq_dim"]),
        stride=int(config["simba_stride"]),
        targeted=targeted,
        batch_size=1,
        verbose=False,
    )


def _build_pixel(estimator, config: dict, targeted: bool):
    """Builds ART's PixelAttack."""
    from art.attacks.evasion import PixelAttack  # type: ignore[import-untyped]
    return PixelAttack(
        estimator,
        th=int(config["pixel_th"]),
        es=int(config["pixel_es"]),
        max_iter=int(config["pixel_max_iter"]),
        targeted=targeted,
        verbose=False,
    )


ATTACKS: Dict[str, AttackSpec] = {
    "square": AttackSpec("square", NutRequirement.SCORES, True,
                         lambda config: _parse_norm(config, "square_norm"), _build_square),
    "hsj": AttackSpec("hsj", NutRequirement.LABEL, False,
                      lambda config: _parse_norm(config, "hsj_norm"), _build_hsj),
    "simba": AttackSpec("simba", NutRequirement.SCORES, True, lambda config: 2, _build_simba),
    "pixel": AttackSpec("pixel", NutRequirement.SCORES, True, lambda config: 0, _build_pixel),
}
