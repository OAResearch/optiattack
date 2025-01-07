import pytest

from core.search.action import Action
from core.search.individual import Individual


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

    # Add the same action again (should not be added)
    assert individual.add_action(action1) is False
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
    assert individual.add_action(action3) is False
    assert individual.size() == 2


def test_add_action_with_different_actions(individual):
    """Test adding different actions."""
    action1 = Action((1, 2), 100, 150, 200)
    action2 = Action((3, 4), 50, 75, 100)
    # Add two different actions
    assert individual.add_action(action1) is True
    assert individual.add_action(action2) is True
    assert individual.size() == 2
    assert individual.get_actions() == [action1, action2]
