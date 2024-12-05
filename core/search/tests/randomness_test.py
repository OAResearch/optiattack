import pytest

from core.config_parser import Config
from core.search.service.randomness import Randomness


@pytest.fixture
def randomness():
    conf = Config()
    rand = Randomness(conf)
    rand.update_seed(42)
    return rand


def test_max_min_int(randomness):
    min_val = -42
    max_val = 1234

    randomness.update_seed(42)
    a = "".join(str(randomness.next_int(min_val, max_val)) for _ in range(100))

    randomness.update_seed(42)
    b = "".join(str(randomness.next_int(min_val, max_val)) for _ in range(100))

    assert a == b