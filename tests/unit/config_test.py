import pytest
from dependency_injector.providers import Configuration

from core.main import OptiAttack
from core.problem.base_module import BaseModule


@pytest.fixture
def container():
    container = BaseModule()
    yield container
    container.unwire()

def test_override_config(container):
    container.config.override({"seed": 42})
    app = OptiAttack(
        config=container.config(),
        randomness=container.randomness(),
    )
    assert app.config.get("seed") == 42

def test_img_path(container):
    container.config.override({"seed": 42, "input_image": "tests/data/image.jpg"})
    app = OptiAttack(
        config=container.config(),
        randomness=container.randomness(),
    )
    assert app.config.get("input_image") == "tests/data/image.jpg"
