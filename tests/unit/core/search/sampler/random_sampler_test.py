import numpy as np
import pytest
from unittest.mock import MagicMock
from core.search.action import Action
from core.search.individual import Individual
from core.search.service.sampler.random_sampler import RandomSampler
from core.search.service.sampler.gaussian_sampler import GaussianSampler
from core.search.service.randomness import Randomness

# Fixture for the RandomSampler instance
@pytest.fixture
def random_sampler():
    randomness = MagicMock()
    archive = MagicMock()
    config = {
        "min_action_size": 1,
        "max_action_size": 5,
        "image_width": 100,
        "image_height": 100
    }
    return RandomSampler(randomness, archive, config)

# Test cases
def test_sample_random_action(random_sampler):
    # Mock the randomness to return specific values
    random_sampler.randomness.next_int.side_effect = [50, 60, 255, 128, 0]

    # Call the sample_random_action method
    action = random_sampler.sample_random_action()

    # Assert that the action has the correct location and color
    assert isinstance(action, Action)
    assert action.get_location() == (50, 60)
    assert all(action.get_color() == [255, 128, 0])
    assert random_sampler.randomness.next_int.call_count == 5

def test_sample(random_sampler):
    # Mock the randomness to return specific values
    random_sampler.randomness.next_int.side_effect = [3, 50, 60, 255, 128, 0, 70, 80, 0, 255, 128, 90, 100, 128, 0, 255]

    # Mock the sample_random_action method
    random_sampler.sample_random_action = MagicMock(side_effect=[
        Action((50, 60), 255, 128, 0),
        Action((70, 80), 0, 255, 128),
        Action((90, 100), 128, 0, 255)
    ])

    # Call the sample method
    individual = random_sampler.sample()

    # Assert that the individual has the correct number of actions
    assert isinstance(individual, Individual)
    assert len(individual.get_actions()) == 3

    # Assert that the actions were added correctly
    actions = individual.get_actions()
    assert actions[0].get_location() == (50, 60)
    assert all(actions[0].get_color() == [255, 128, 0])
    assert actions[1].get_location() == (70, 80)
    assert all(actions[1].get_color() == [0, 255, 128])
    assert actions[2].get_location() == (90, 100)
    assert all(actions[2].get_color() == [128, 0, 255])

def test_sample_with_retry(random_sampler):
    # Mock the randomness to return specific values
    random_sampler.randomness.next_int.side_effect = [2, 50, 60, 255, 128, 0, 50, 60, 0, 255, 128, 70, 80, 128, 0, 255]

    # Mock the sample_random_action method to return duplicate actions initially
    random_sampler.sample_random_action = MagicMock(side_effect=[
        Action((50, 60), 255, 128, 0),  # Duplicate action
        Action((50, 60), 0, 255, 128),  # Duplicate action
        Action((70, 80), 128, 0, 255)   # Unique action
    ])

    # Call the sample method
    individual = random_sampler.sample()

    # Assert that the individual has the correct number of actions
    assert isinstance(individual, Individual)
    assert len(individual.get_actions()) == 2

    # Assert that the actions were added correctly
    actions = individual.get_actions()
    assert actions[0].get_location() == (50, 60)
    assert all(actions[0].get_color() == [255, 128, 0])
    assert actions[1].get_location() == (70, 80)
    assert all(actions[1].get_color() == [128, 0, 255])


def test_sample_random_action_uses_row_col_on_non_square_image():
    # Locations are (row, col): row < image_height, col < image_width
    randomness = MagicMock()
    randomness.next_int.side_effect = lambda low, high: high - 1
    sampler = RandomSampler(randomness, MagicMock(), {"image_width": 30, "image_height": 10})

    action = sampler.sample_random_action()

    assert action.get_location() == (9, 29)


@pytest.mark.parametrize("sampler_class", [RandomSampler, GaussianSampler])
def test_sample_size_covers_min_and_max_action_size(sampler_class):
    # max_action_size is inclusive, as documented in docs/parameters.md
    config = {"seed": 3, "min_action_size": 1, "max_action_size": 2, "image_width": 16, "image_height": 16,
              "mutation_sigma": 10}
    archive = MagicMock()
    archive.image.array = np.zeros((16, 16, 3), dtype=np.uint8)
    sampler = sampler_class(Randomness(config), archive, config)

    sizes = {sampler.sample().size() for _ in range(200)}

    assert sizes == {1, 2}
