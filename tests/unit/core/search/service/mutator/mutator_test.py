import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from core.search.action import Action
from core.search.individual import Individual
from core.search.service.mutator.standard_mutator import StandardMutator


@pytest.fixture
def standard_mutator():
    # Mock the required arguments for Mutator
    randomness = MagicMock()
    time = MagicMock()
    config = {"mutation_sigma": 0.1}

    # Initialize StandardMutator with the required arguments
    mutator = StandardMutator(randomness, time, config)
    return mutator


# Test cases
def test_mutate_with_empty_actions(standard_mutator):
    """Test mutate method with an individual that has no actions."""
    individual = Individual()
    result = standard_mutator.mutate(individual)
    assert result == individual


def test_mutate_with_non_empty_actions(standard_mutator):
    """Test mutate method with an individual that has actions."""
    # Create actions and individual
    action = Action((1, 2), 100, 150, 200)
    individual = Individual()
    individual.add_action(action)

    # Mock the randomness to return a specific index
    standard_mutator.randomness.next_int.return_value = 0

    # Mock the Gaussian mutation to return a specific value
    mutated_action = Action((1, 2), 110, 160, 210)
    standard_mutator.apply_gaussian_mutation = MagicMock(return_value=mutated_action)

    result = standard_mutator.mutate(individual)

    # Assert that the action was mutated
    assert result.get_actions()[0].get_color() == pytest.approx([110, 160, 210])
    standard_mutator.randomness.next_int.assert_called_once_with(0, 0)
    standard_mutator.apply_gaussian_mutation.assert_called_once_with(action)


def test_apply_gaussian_mutation(standard_mutator):
    """Test apply_gaussian_mutation method."""
    # Create an action
    action = Action((1, 2), 100, 150, 200)

    # Mock the Gaussian random value
    standard_mutator.randomness.random_gaussian.return_value = [10, 10, 10]

    result = standard_mutator.apply_gaussian_mutation(action)

    # Assert that the color was mutated correctly
    assert result.get_color() == pytest.approx([110, 160, 210])
    standard_mutator.randomness.random_gaussian.assert_called_once()
    call_args = standard_mutator.randomness.random_gaussian.call_args
    assert np.array_equal(call_args[0][0], np.array([0, 0, 0]))  # Check mean
    assert call_args[0][1] == 0.1  # Check sigma


def test_apply_gaussian_mutation_with_limit_values(standard_mutator):
    """Test apply_gaussian_mutation method with values exceeding limits."""
    # Create an action
    action = Action((1, 2), 100, 150, 200)

    # Mock the Gaussian random value to exceed limits
    standard_mutator.randomness.random_gaussian.return_value = [300, 300, 300]

    result = standard_mutator.apply_gaussian_mutation(action)

    # Assert that the color was clamped to the limit values
    assert result.get_color() == pytest.approx([255, 255, 255])
