from array import array

import numpy as np
import pytest

from core.search.action import Action
from core.search.individual import Individual
from core.utils.images import ProcessedImage


@pytest.fixture
def individual():
    """Fixture to create an Individual instance for testing."""
    return Individual()


def test_initialization(individual):
    """Test that the Individual class initializes correctly."""
    assert individual.actions == []
    assert individual.individual_origin is None


def test_add_action(individual):
    """Test the add_action method."""
    action1 = Action((1, 2), 100, 150, 200)
    action2 = Action((3, 4), 50, 75, 100)
    assert individual.add_action(action1) is True
    assert action1 in individual.actions

    # Add the same action again (just replace it)
    assert individual.add_action(action1) is True
    assert len(individual.actions) == 1

    # Add a different action
    assert individual.add_action(action2) is True
    assert action2 in individual.actions
    assert len(individual.actions) == 2


def test_get_actions(individual):
    """Test the get_actions method."""
    action1 = Action((1, 2), 100, 150, 200)
    action2 = Action((3, 4), 50, 75, 100)

    # Initially, actions should be empty
    assert individual.get_actions() == []

    # Add actions and check the result
    individual.add_action(action1)
    individual.add_action(action2)
    assert individual.get_actions() == [action1, action2]


def test_set_individual_origin(individual):
    """Test the set_individual_origin method."""
    origin = "Origin1"
    individual.set_individual_origin(origin)
    assert individual.individual_origin == origin


def test_size(individual):
    """Test the size method."""
    action1 = Action((1, 2), 100, 150, 200)
    action2 = Action((3, 4), 50, 75, 100)
    # Initially, size should be 0
    assert individual.size() == 0

    # Add actions and check the size
    individual.add_action(action1)
    assert individual.size() == 1

    individual.add_action(action2)
    assert individual.size() == 2


def test_add_action_with_duplicates(individual):
    """Test adding duplicate actions."""
    action1 = Action((1, 2), 100, 150, 200)
    action2 = Action((3, 4), 50, 75, 100)
    action3 = Action((1, 2), 100, 150, 200)

    # Add the same action twice
    assert individual.add_action(action1) is True
    assert individual.add_action(action2) is True
    assert individual.add_action(action3) is True
    assert individual.size() == 2
    assert individual.get_actions() == [action3, action2]


def test_add_action_with_different_actions(individual):
    """Test adding different actions."""
    action1 = Action((1, 2), 100, 150, 200)
    action2 = Action((3, 4), 50, 75, 100)
    # Add two different actions
    assert individual.add_action(action1) is True
    assert individual.add_action(action2) is True
    assert individual.size() == 2
    assert individual.get_actions() == [action1, action2]

def test_get_action_image_no_actions(individual):
    # Create a mock image with a 3x3 array
    image_array = np.zeros((3, 3, 3), dtype=np.uint8)  # 3x3 RGB image

    # Call the function
    result = individual.get_action_image(image_array)

    # Assert that the image is unchanged
    assert np.array_equal(result, image_array)

def test_get_action_image_with_actions(individual):
    # Create a mock image with a 3x3 array
    image_array = np.zeros((3, 3, 3), dtype=np.uint8)  # 3x3 RGB image

    action1 = Action((1, 1), 255, 255, 255)  # White pixel at (1, 1)
    action2 = Action((0, 0), 255, 0, 0)  # Red pixel at (0, 0)

    # Create an individual with actions
    individual.add_action(action1)
    individual.add_action(action2)

    # Call the function
    result = individual.get_action_image(image_array)

    # Expected result: image with modified pixels
    expected_image = image_array.copy()
    expected_image[1, 1] = [255, 255, 255]  # Pixel at (1, 1) is white
    expected_image[0, 0] = [255, 0, 0]  # Pixel at (0, 0) is red

    # Assert that the image is modified correctly
    assert np.array_equal(result, expected_image)

def test_get_action_image_multiple_actions_same_location(individual):
    # Create a mock image with a 3x3 array
    image_array = np.zeros((3, 3, 3), dtype=np.uint8)  # 3x3 RGB image

    # Create actions with the same location
    action1 = Action((1, 1), 255, 255, 255)  # White pixel at (1, 1)
    action2 = Action((1, 1), 0, 0, 255)  # Blue pixel at (1, 1)

    # Create an individual with actions
    individual.add_action(action1)
    individual.add_action(action2)

    assert individual.get_actions() == [action2]
    assert individual.size() == 1

    # Call the function
    result = individual.get_action_image(image_array)

    # Expected result: the last action should overwrite the previous one
    expected_image = image_array.copy()
    expected_image[1, 1] = [0, 0, 255]  # Pixel at (1, 1) is blue

    # Assert that the image is modified correctly
    assert np.array_equal(result, expected_image)

def test_individual_crossover():
    individual = Individual()
    other = Individual()
    action1 = Action((1, 2), 100, 150, 200)
    action2 = Action((3, 4), 50, 75, 100)
    action3 = Action((5, 6), 200, 100, 50)
    action4 = Action((7, 8), 255, 0, 0)
    action5 = Action((9, 10), 255, 0, 0)
    action6 = Action((11, 12), 0, 255, 0)

    individual.add_action(action1)
    individual.add_action(action2)
    individual.add_action(action3)
    other.add_action(action4)
    other.add_action(action5)
    other.add_action(action6)

    assert individual.size() == 3
    assert other.size() == 3

    position1 = 1
    position2 = 2

    individual.crossover(other, position1, position2)
    assert individual.size() == 2

