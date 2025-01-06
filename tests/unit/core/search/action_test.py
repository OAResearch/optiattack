import pytest
import numpy as np
from core.search.action import Action


class TestAction:
    @pytest.fixture
    def action(self):
        """Fixture to create an Action instance for testing."""
        return Action((1, 2), 100, 150, 200)

    def test_initialization(self, action):
        """Test that the Action class initializes correctly."""
        assert action.location == (1, 2)
        assert action.red == 100
        assert action.green == 150
        assert action.blue == 200
        assert action.parent is None
        assert action.noise == 0

    def test_set_parent(self, action):
        """Test the set_parent method."""
        parent_action = Action((0, 0), 0, 0, 0)
        action.set_parent(parent_action)
        assert action.get_parent() == parent_action

    def test_get_parent(self, action):
        """Test the get_parent method."""
        assert action.get_parent() is None  # Initially, parent should be None
        parent_action = Action((0, 0), 0, 0, 0)
        action.set_parent(parent_action)
        assert action.get_parent() == parent_action

    def test_get_location(self, action):
        """Test the get_location method."""
        assert action.get_location() == (1, 2)

    def test_color(self, action):
        """Test the color method."""
        expected_color = np.array([100, 150, 200])
        assert np.array_equal(action.color(), expected_color)
