import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from core.search.action import Action
from core.search.individual import Individual
from core.search.phase_controller import PhaseController
from core.search.service.adaptive_parameter_control import AdaptiveParameterControl
from core.search.service.mutator.standard_mutator import StandardMutator
from core.search.service.randomness import Randomness
from core.search.service.search_time_controller import SearchTimeController
from tests.utils import run_multiple


@pytest.fixture
def standard_mutator():
    # Mock the required arguments for Mutator
    time = MagicMock()
    config = {"seed":42, "image_height": 224, "image_width": 224, "mutation_sigma": 0.1, "apc_pixel_start": 0, "apc_pixel_end": 255, "start_time": 0.5, "threshold": 1.0}
    randomness = Randomness(config)

    stc = SearchTimeController(config, pc=PhaseController())
    apc = AdaptiveParameterControl(stc, config)
    # Initialize StandardMutator with the required arguments
    mutator = StandardMutator(randomness, time, config, apc)
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
    standard_mutator.randomness.next_int = MagicMock(return_value=0)

    # Mock the Gaussian mutation to return a specific value
    mutated_action = Action((1, 2), 110, 160, 210)
    standard_mutator.apply_gaussian_mutation = MagicMock(return_value=mutated_action)

    result = standard_mutator.mutate(individual)

    # Assert that the action was mutated
    assert result.get_actions()[0].get_color() == pytest.approx([110, 160, 210])
    standard_mutator.apply_gaussian_mutation.assert_called_once_with(action)

@run_multiple()
def test_apply_gaussian_mutation_zero_std(standard_mutator):
    """Test apply_gaussian_mutation method."""
    # Create an action
    action = Action((1, 2), 100, 150, 200)

    # zero std should be return always the mean value ([0, 0, 0])
    standard_mutator.apc.get_pixel_apc = MagicMock(return_value=0)
    standard_mutator.apc.get_location_apc = MagicMock(return_value=0)

    result = standard_mutator.apply_gaussian_mutation(action)

    # Assert that the color was mutated correctly
    assert result.get_color() == pytest.approx([100, 150, 200], abs=0.1)

#TODO will handle this in the future
def not_test_apply_gaussian_mutation_with_limit_values(standard_mutator):
    """Test apply_gaussian_mutation method with values exceeding limits."""
    # Create an action
    action = Action((1, 2), 100, 150, 200)

    # Mock the Gaussian random value to exceed limits
    standard_mutator.apc.get_pixel_apc = MagicMock(return_value=[300, 300, 300])
    standard_mutator.apc.get_location_apc = MagicMock(return_value=[20, 20])

    result = standard_mutator.apply_gaussian_mutation(action)

    # Assert that the color was clamped to the limit values
    assert result.get_color() == pytest.approx([255, 255, 255])
