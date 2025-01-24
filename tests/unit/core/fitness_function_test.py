import pytest
from unittest.mock import MagicMock
from core.search.fitness_value import FitnessValue
from core.search.individual import Individual
from core.search.service.fitness_function import FitnessFunction


# Mock classes for dependencies
class MockIndividual(Individual):
    def get_action_image(self, image):
        return "mock_image_array"

class MockArchive:
    def get_image(self):
        return "mock_image"

    def get_original_prediction_results(self):
        return MagicMock(max_score=MagicMock(label="original_label"))

class MockRemoteController:
    def new_action(self, img_array):
        return MagicMock(
            max_score=MagicMock(value=0.9, label="original_label"),
            second_max_score=MagicMock(value=0.1)
        )

# Fixture for the FitnessFunction instance
@pytest.fixture
def fitness_function():
    archive = MockArchive()
    remote_controller = MockRemoteController()
    return FitnessFunction(archive, remote_controller)

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
