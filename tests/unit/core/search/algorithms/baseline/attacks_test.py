from unittest.mock import MagicMock

import numpy as np
import pytest

from core.config_parser import ConfigParser
from core.search.algorithms.baseline.attacks import ATTACKS, AttackSpec, resolve_stop_at_success
from core.search.algorithms.baseline.nut_interface import NutRequirement

LABELS = ["dark", "bright", "mid"]
SHAPE = (16, 24, 3)


def make_config(**overrides):
    config = dict(ConfigParser().default_params())
    config.update(overrides)
    return config


def make_estimator():
    pytest.importorskip("art")
    from core.search.algorithms.baseline.estimator import RemoteNutEstimator
    return RemoteNutEstimator(MagicMock(), LABELS, SHAPE, MagicMock())


def test_all_attacks_are_registered():
    assert set(ATTACKS) == {"square", "hsj", "simba", "pixel"}
    assert all(isinstance(spec, AttackSpec) for spec in ATTACKS.values())
    assert all(spec.name == name for name, spec in ATTACKS.items())


def test_requirements():
    assert ATTACKS["square"].requirement == NutRequirement.SCORES
    assert ATTACKS["simba"].requirement == NutRequirement.SCORES
    assert ATTACKS["pixel"].requirement == NutRequirement.SCORES
    assert ATTACKS["hsj"].requirement == NutRequirement.LABEL


def test_defaults_match_the_design():
    config = make_config()
    assert config["baseline_attack"] == "square"
    assert config["baseline_stop_at_success"] == "auto"
    assert config["square_norm"] == "inf"
    assert config["square_eps"] == 8.0
    assert config["hsj_norm"] == "2"
    assert config["hsj_init_image"] == ""
    assert config["simba_epsilon"] == 25.5
    assert config["pixel_th"] == 10


@pytest.mark.parametrize("name, expected", [("square", True), ("simba", True), ("pixel", True), ("hsj", False)])
def test_stop_at_success_auto_uses_attack_default(name, expected):
    assert resolve_stop_at_success(ATTACKS[name], make_config(baseline_stop_at_success="auto")) is expected


@pytest.mark.parametrize("name", ["square", "hsj", "simba", "pixel"])
def test_stop_at_success_can_be_forced(name):
    assert resolve_stop_at_success(ATTACKS[name], make_config(baseline_stop_at_success="true")) is True
    assert resolve_stop_at_success(ATTACKS[name], make_config(baseline_stop_at_success="FALSE")) is False


def test_stop_at_success_rejects_unknown_value():
    with pytest.raises(ValueError, match="baseline_stop_at_success"):
        resolve_stop_at_success(ATTACKS["square"], make_config(baseline_stop_at_success="maybe"))


def test_norms():
    assert ATTACKS["square"].norm(make_config(square_norm="inf")) == np.inf
    assert ATTACKS["square"].norm(make_config(square_norm="2")) == 2
    assert ATTACKS["hsj"].norm(make_config(hsj_norm="2")) == 2
    assert ATTACKS["hsj"].norm(make_config(hsj_norm="inf")) == np.inf
    assert ATTACKS["simba"].norm(make_config()) == 2
    assert ATTACKS["pixel"].norm(make_config()) == 0


def test_invalid_norm_raises():
    with pytest.raises(ValueError, match="square_norm"):
        ATTACKS["square"].norm(make_config(square_norm="1"))


def test_build_square():
    estimator = make_estimator()
    from art.attacks.evasion import SquareAttack
    attack = ATTACKS["square"].build(estimator, make_config(square_eps=4.0, square_p_init=0.5, square_nb_restarts=2,
                                                            square_max_iter=77, square_norm="2"), False)
    assert isinstance(attack, SquareAttack)
    assert attack.eps == 4.0
    assert attack.p_init == 0.5
    assert attack.nb_restarts == 2
    assert attack.max_iter == 77
    assert attack.norm == 2
    assert attack.estimator is estimator


def test_square_adv_criterion_is_always_false():
    estimator = make_estimator()
    for targeted in (False, True):
        attack = ATTACKS["square"].build(estimator, make_config(), targeted)
        y_pred = np.array([[0.1, 0.9, 0.0], [0.9, 0.1, 0.0]])
        y = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert not np.any(attack.adv_criterion(y_pred, y))
        assert np.shape(attack.adv_criterion(y_pred, y)) == (2,)


def test_square_targeted_loss_is_best_other_minus_target():
    estimator = make_estimator()
    estimator.predict = MagicMock(return_value=np.array([[0.5, 0.2, 0.3], [0.1, 0.6, 0.3]]))
    attack = ATTACKS["square"].build(estimator, make_config(), True)
    y = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    loss = attack.loss(np.zeros((2,) + SHAPE), y)
    np.testing.assert_allclose(loss, [0.5 - 0.2, 0.3 - 0.6])


def test_build_hsj():
    estimator = make_estimator()
    from art.attacks.evasion import HopSkipJump
    config = make_config(hsj_norm="inf", hsj_max_iter=7, hsj_max_eval=300, hsj_init_eval=20, hsj_init_size=30)
    attack = ATTACKS["hsj"].build(estimator, config, True)
    assert isinstance(attack, HopSkipJump)
    assert attack.norm == np.inf
    assert attack.max_iter == 7
    assert attack.max_eval == 300
    assert attack.init_eval == 20
    assert attack.init_size == 30
    assert attack.targeted is True
    assert ATTACKS["hsj"].build(estimator, make_config(), False).targeted is False


def test_build_simba():
    estimator = make_estimator()
    from art.attacks.evasion import SimBA
    config = make_config(simba_attack="px", simba_epsilon=12.0, simba_max_iter=99, simba_order="random",
                         simba_freq_dim=3, simba_stride=2)
    attack = ATTACKS["simba"].build(estimator, config, True)
    assert isinstance(attack, SimBA)
    assert attack.attack == "px"
    assert attack.epsilon == 12.0
    assert attack.max_iter == 99
    assert attack.order == "random"
    assert attack.freq_dim == 3
    assert attack.stride == 2
    assert attack.targeted is True
    assert ATTACKS["simba"].build(estimator, make_config(simba_attack="px"), False).targeted is False


def test_build_pixel():
    estimator = make_estimator()
    from art.attacks.evasion import PixelAttack
    attack = ATTACKS["pixel"].build(estimator, make_config(pixel_th=5, pixel_es=0, pixel_max_iter=12), True)
    assert isinstance(attack, PixelAttack)
    assert attack.th == 5
    assert attack.es == 0
    assert attack.max_iter == 12
    assert attack.targeted is True
    assert ATTACKS["pixel"].build(estimator, make_config(), False).targeted is False


@pytest.mark.parametrize("overrides", [{"simba_attack": "dct"}, {"simba_attack": "px", "simba_order": "diag"}])
def test_simba_square_only_modes_reject_non_square_images(overrides):
    estimator = make_estimator()
    with pytest.raises(ValueError, match="simba_attack px"):
        ATTACKS["simba"].build(estimator, make_config(**overrides), False)


def test_simba_px_random_order_accepts_non_square_images():
    estimator = make_estimator()
    attack = ATTACKS["simba"].build(estimator, make_config(simba_attack="px", simba_order="random"), False)
    assert attack.attack == "px"


def test_square_size_is_capped_below_the_shorter_side():
    estimator = make_estimator()
    attack = ATTACKS["square"].build(estimator, make_config(square_p_init=0.8), False)
    p = attack._get_percentage_of_elements(0)
    assert round((p * SHAPE[0] * SHAPE[1]) ** 0.5) <= SHAPE[0] - 1


def make_estimator_for(shape):
    pytest.importorskip("art")
    from core.search.algorithms.baseline.estimator import RemoteNutEstimator
    return RemoteNutEstimator(MagicMock(), LABELS, shape, MagicMock())


@pytest.mark.parametrize("hw", [(17, 30), (16, 24), (5, 9), (5, 5), (20, 20)])
def test_square_l2_tile_fits_with_odd_adjustment(hw):
    attack = ATTACKS["square"].build(make_estimator_for(hw + (3,)), make_config(square_norm="2"), False)
    height, width = hw
    for i_iter in (0, 1, attack.max_iter - 1):
        p = attack._get_percentage_of_elements(i_iter)
        # ART's L2 branch: tile = max(round(sqrt(p*h*w)), 3), bumped to the next odd number.
        tile = max(int(round((p * height * width) ** 0.5)), 3)
        tile += 1 if tile % 2 == 0 else 0
        assert tile <= min(height, width) - 1


@pytest.mark.parametrize("norm, hw", [("2", (3, 10)), ("2", (4, 10)), ("2", (10, 2)), ("inf", (1, 10)), ("inf", (10, 1))])
def test_square_rejects_images_too_small_for_any_tile(norm, hw):
    with pytest.raises(ValueError, match="at least"):
        ATTACKS["square"].build(make_estimator_for(hw + (3,)), make_config(square_norm=norm), False)


@pytest.mark.parametrize("hw", [(30, 17), (224, 200)])
def test_square_l2_rejects_tall_images_that_break_art_initialisation(hw):
    # ART's L2 start lays 5x5 tiles of side height // 5 along both axes, so the width must hold 5 of them.
    with pytest.raises(ValueError, match="square_norm inf"):
        ATTACKS["square"].build(make_estimator_for(hw + (3,)), make_config(square_norm="2"), False)


def test_square_inf_accepts_tall_images():
    attack = ATTACKS["square"].build(make_estimator_for((30, 17, 3)), make_config(square_norm="inf"), False)
    assert attack.norm == np.inf


def test_square_max_iter_auto_uses_the_evaluation_budget():
    config = make_config(square_max_iter=0, max_evaluations=1234,
                         stopping_criterion=ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS)
    assert ATTACKS["square"].build(make_estimator(), config, False).max_iter == 1234


def test_square_max_iter_auto_with_time_budget_is_10000():
    config = make_config(square_max_iter=0, max_evaluations=60, stopping_criterion=ConfigParser.StoppingCriterion.TIME)
    assert ATTACKS["square"].build(make_estimator(), config, False).max_iter == 10000


def test_square_max_iter_default_is_auto():
    assert make_config()["square_max_iter"] == 0
