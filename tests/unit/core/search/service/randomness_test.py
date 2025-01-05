import pytest

from core.main import OptiAttack
from core.problem.base_module import BaseModule


@pytest.fixture
def app():
    container = BaseModule()
    container.unwire()
    container.config.override({"seed": 42})

    app = OptiAttack(
        config=container.config(),
        randomness=container.randomness(),
    )
    yield app


def test_max_min_int(app):
    min_val = -42
    max_val = 1234

    app.randomness.update_seed(42)
    a = "".join(str(app.randomness.next_int(min_val, max_val)) for _ in range(100))

    app.randomness.update_seed(42)
    b = "".join(str(app.randomness.next_int(min_val, max_val)) for _ in range(100))

    assert a == b

def test_max_min_float(app):
    min_val = -42.0
    max_val = 1234.0

    app.randomness.update_seed(42)
    a = "".join(str(app.randomness.next_float(min_val, max_val)) for _ in range(100))

    app.randomness.update_seed(42)
    b = "".join(str(app.randomness.next_float(min_val, max_val)) for _ in range(100))

    assert a == b

def test_choice(app):
    app.randomness.update_seed(42)
    a = "".join(str(app.randomness.random_choice(5)) for _ in range(100))

    app.randomness.update_seed(42)
    b = "".join(str(app.randomness.random_choice(5)) for _ in range(100))

    assert a == b

def test_random_gaussian(app):
    mean = 0
    std = 1

    app.randomness.update_seed(42)
    a = "".join(str(app.randomness.random_gaussian(mean, std)) for _ in range(100))

    app.randomness.update_seed(42)
    b = "".join(str(app.randomness.random_gaussian(mean, std)) for _ in range(100))

    assert a == b
