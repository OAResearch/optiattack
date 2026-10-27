import logging

import numpy as np
import pytest

from core.config_parser import ConfigParser
from tests.unit.core.search.algorithms.baseline.baseline_algorithm_test import (
    ATTACK_NAMES, make_algorithm)

SIZE = 224
THRESHOLD = 128.0
BUDGET = 30
MIN_QUERIES = 5


def gradient_image():
    """A smooth 224x224x3 gradient image with a mean of about 100."""
    ramp = np.linspace(60, 140, SIZE)
    image = np.stack([np.add.outer(ramp, ramp) / 2.0] * 3, axis=-1)
    return np.clip(image, 0, 255).astype(np.uint8)


def far_brightness_scores(image):
    """Brightness scores whose class flips only for images with a mean near THRESHOLD, e.g. uniform noise.

    The smooth original has a mean of about 100, so small or local perturbations (SquareAttack stripes, a few
    pixels) never flip it within 30 queries, while HopSkipJump's random-noise start image (mean about 127.5)
    can, which lets HSJ initialise instead of stalling.
    """
    mean = float(np.mean(image))
    bright = 1.0 / (1.0 + np.exp(-(mean - THRESHOLD) / 2.0))
    return {"dark": 0.7 * (1.0 - bright), "bright": bright, "mid": 0.3 * (1.0 - bright)}


def default_overrides(attack):
    """Config overrides that keep the attack's default parameters (make_algorithm forces simba to 'px')."""
    if attack == "simba":
        return {"simba_attack": ConfigParser().default_params()["simba_attack"]}
    return {}


@pytest.fixture
def art():
    return pytest.importorskip("art")


@pytest.mark.parametrize("attack", ATTACK_NAMES)
def test_attack_makes_real_queries_on_224_images(art, attack, caplog):
    algorithm, nut = make_algorithm(attack, max_evaluations=BUDGET, scores_fn=far_brightness_scores,
                                    image=gradient_image(), **default_overrides(attack))
    with caplog.at_level(logging.WARNING):
        algorithm.search()
    assert len(nut.images) >= MIN_QUERIES
    assert "was stopped" not in caplog.text


def test_simba_dct_on_224_images_stalls_with_hint(art, caplog):
    algorithm, nut = make_algorithm("simba", max_evaluations=BUDGET, scores_fn=far_brightness_scores,
                                    image=gradient_image(), simba_attack="dct")
    with caplog.at_level(logging.WARNING):
        algorithm.search()
    assert "was stopped: it produced no new image" in caplog.text
    assert "simba_epsilon" in caplog.text
