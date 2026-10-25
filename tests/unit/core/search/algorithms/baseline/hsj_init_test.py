import logging

import numpy as np
import pytest
from PIL import Image

from core.search.algorithms.baseline.hsj_init import find_hsj_initial_image
from core.search.service.randomness import Randomness

HEIGHT, WIDTH = 8, 10


class FakeEstimator:

    """Counts predictions; class 1 when the mean is above 128, class 0 otherwise."""

    def __init__(self):
        self.images = []

    def predict(self, x, batch_size=128, **kwargs):
        out = []
        for image in x:
            self.images.append(np.array(image, copy=True))
            bright = float(np.mean(image)) > 128
            out.append([0.0, 1.0] if bright else [1.0, 0.0])
        return np.asarray(out, dtype=np.float32)


def is_bright(vector):
    return int(np.argmax(vector)) == 1


def never(vector):
    return False


def make_config(**overrides):
    config = {"seed": 3, "image_width": WIDTH, "image_height": HEIGHT, "hsj_init_size": 50, "hsj_init_image": ""}
    config.update(overrides)
    return config


def original():
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def save_image(path, value):
    Image.fromarray(np.full((HEIGHT + 3, WIDTH + 5, 3), value, dtype=np.uint8)).save(str(path))


def test_uses_given_init_image_when_adversarial(tmp_path):
    path = tmp_path / "init.png"
    save_image(path, 250)
    estimator = FakeEstimator()
    config = make_config(hsj_init_image=str(path))
    init = find_hsj_initial_image(estimator, original(), Randomness(config), config, is_bright)
    assert init.shape == (HEIGHT, WIDTH, 3)
    assert init.dtype == np.float32
    assert np.allclose(init, 250)
    assert len(estimator.images) == 1


def test_missing_init_image_file_warns_and_falls_back(tmp_path, caplog):
    estimator = FakeEstimator()
    config = make_config(hsj_init_image=str(tmp_path / "missing.png"))
    with caplog.at_level(logging.WARNING):
        init = find_hsj_initial_image(estimator, original(), Randomness(config), config, is_bright)
    assert "missing.png" in caplog.text
    assert init is not None and is_bright(estimator.predict(init[None])[0])


def test_non_adversarial_init_image_warns_and_falls_back(tmp_path, caplog):
    path = tmp_path / "init.png"
    save_image(path, 10)
    estimator = FakeEstimator()
    config = make_config(hsj_init_image=str(path))
    with caplog.at_level(logging.WARNING):
        init = find_hsj_initial_image(estimator, original(), Randomness(config), config, is_bright)
    assert "not adversarial" in caplog.text
    assert init is not None and float(np.mean(init)) > 128
    assert len(estimator.images) >= 2


def test_random_init_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        estimator = FakeEstimator()
        config = make_config()
        init = find_hsj_initial_image(estimator, original(), Randomness(config), config, is_bright)
        results.append((init, len(estimator.images)))
    assert np.array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]
    assert results[0][0].dtype == np.float32
    assert results[0][0].min() >= 0 and results[0][0].max() <= 255


def test_returns_none_when_no_adversarial_found_within_init_size():
    estimator = FakeEstimator()
    config = make_config(hsj_init_size=7)
    assert find_hsj_initial_image(estimator, original(), Randomness(config), config, never) is None
    assert len(estimator.images) == 7


def test_stop_attack_propagates():
    from core.search.algorithms.baseline.query_tracker import StopAttack

    class Stopping(FakeEstimator):
        def predict(self, x, batch_size=128, **kwargs):
            raise StopAttack()

    config = make_config()
    with pytest.raises(StopAttack):
        find_hsj_initial_image(Stopping(), original(), Randomness(config), config, is_bright)
