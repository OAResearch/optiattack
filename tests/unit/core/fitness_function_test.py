import pytest
from unittest.mock import MagicMock

from core.remote.remote_controller import RemoteController
from core.search.fitness_value import FitnessValue
from core.search.individual import Individual
from core.search.service.archive import Archive
from core.search.service.fitness_function import FitnessFunction
from core.search.service.randomness import Randomness
from core.search.service.search_time_controller import SearchTimeController


# Mock classes for dependencies
class MockIndividual(Individual):
    def get_action_image(self, image):
        return "mock_image_array"

class MockArchive(Archive):
    def get_mutated_image(self):
        return "mock_image"

    def get_original_prediction_results(self):
        return MagicMock(max_score=MagicMock(label="original_label"))

class MockRemoteController(RemoteController):
    def new_action(self, img_array):
        return MagicMock(
            max_score=MagicMock(value=0.9, label="original_label"),
            second_max_score=MagicMock(value=0.1)
        )

# Fixture for the FitnessFunction instance
@pytest.fixture
def fitness_function():
    stc = MagicMock(spec=SearchTimeController)
    config = {}
    randomness = MagicMock(spec=Randomness)
    archive = MockArchive(stc, randomness, config)
    remote_controller = MockRemoteController(config, stc)
    stc = MagicMock(spec=SearchTimeController)
    return FitnessFunction(archive, remote_controller, stc)

# Test cases
def test_evaluate_with_matching_labels(fitness_function):
    # Create a mock individual
    individual = MockIndividual()

    # Call the evaluate method
    result = fitness_function.evaluate(individual)

    # Assert that the fitness value is calculated correctly
    assert isinstance(result, FitnessValue)
    assert result.value == 0.8  # 0.9 (max_score) - 0.1 (second_max_score)

def test_evaluate_with_non_matching_labels(fitness_function):
    # Modify the remote controller to return a non-matching label
    fitness_function.remote_controller.new_action = MagicMock()
    fitness_function.remote_controller.new_action.return_value = MagicMock(
        max_score=MagicMock(value=0.9, label="different_label"),
        second_max_score=MagicMock(value=0.1)
    )

    # Create a mock individual
    individual = MockIndividual()

    # Call the evaluate method
    result = fitness_function.evaluate(individual)

    # Assert that the fitness value is 0 (labels do not match)
    assert isinstance(result, FitnessValue)
    assert result.value == 0
