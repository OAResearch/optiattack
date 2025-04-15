import pytest
from unittest.mock import MagicMock

from core.search.service.fitness_function.targeted_fitness_function import TargetedFitnessFunction
from core.remote.remote_controller import RemoteController
from core.search.fitness_value import FitnessValue
from core.search.individual import Individual
from core.search.service.archive import Archive
from core.search.service.randomness import Randomness
from core.search.service.search_time_controller import SearchTimeController


class MockIndividual(Individual):
    def get_action_image(self, image):
        return "mock_image_array"


class MockArchive(Archive):
    def get_mutated_image(self, actions=None):
        return "mock_image"

    def get_original_prediction_results(self):
        return MagicMock(max_score=MagicMock(label="original_label"))


class MockRemoteController(RemoteController):
    def new_action(self, img_array, target=None):
        return MagicMock(
            max_score=MagicMock(value=0.9, label="original_label"),
            targeted_score=MagicMock(value=0.2),
            predictions=["original_label", "coyote"]
        )


# Fixture for the TargetedFitnessFunction instance
@pytest.fixture
def fitness_function():
    stc = MagicMock(spec=SearchTimeController)
    config = {}
    randomness = MagicMock(spec=Randomness)
    archive = MockArchive(stc, randomness, config)
    remote_controller = MockRemoteController(config, stc)
    target = "coyote"
    return TargetedFitnessFunction(archive, remote_controller, stc, target)


# Test cases
def test_evaluate_with_matching_labels(fitness_function):
    # Create a mock individual
    individual = MockIndividual()

    # Call the evaluate method
    result = fitness_function.evaluate(individual)

    # Assert that the fitness value is calculated correctly
    assert isinstance(result, FitnessValue)
    assert result.value == 0.7  # 0.9 (max_score) - 0.2 (targeted_score)
    assert result.predictions == ["original_label", "coyote"]


def test_evaluate_with_actions_no_individual(fitness_function):
    # Call the evaluate method with actions only
    actions = ["mock_action1", "mock_action2"]
    result = fitness_function.evaluate(actions=actions)

    # Assert that the fitness value is calculated correctly
    assert isinstance(result, FitnessValue)
    assert result.value == 0.7  # 0.9 (max_score) - 0.2 (targeted_score)
    assert result.predictions == ["original_label", "coyote"]


def test_evaluate_with_non_matching_labels(fitness_function):
    # Modify the remote controller to return a non-matching label
    fitness_function.remote_controller.new_action = MagicMock()
    fitness_function.remote_controller.new_action.return_value = MagicMock(
        max_score=MagicMock(value=0.9, label="different_label"),
        targeted_score=MagicMock(value=0.2),
        predictions=["different_label", "coyote"]
    )

    # Create a mock individual
    individual = MockIndividual()

    # Call the evaluate method
    result = fitness_function.evaluate(individual)

    # Assert that the fitness value is 0 (labels do not match)
    assert isinstance(result, FitnessValue)
    assert result.value == 0
    assert result.predictions == ["different_label", "coyote"]


def test_target_initialization(fitness_function):
    # Test that the target is correctly initialized
    assert fitness_function.target == "coyote"


def test_evaluate_passes_target_to_remote_controller(fitness_function):
    # Create a spy on the remote_controller.new_action method
    spy_new_action = MagicMock()
    fitness_function.remote_controller.new_action = spy_new_action

    # Set up return value
    spy_new_action.return_value = MagicMock(
        max_score=MagicMock(value=0.9, label="original_label"),
        targeted_score=MagicMock(value=0.2),
        predictions=["original_label", "coyote"]
    )

    # Call evaluate
    individual = MockIndividual()
    fitness_function.evaluate(individual)

    # Assert that new_action was called with the target
    spy_new_action.assert_called_once_with("mock_image_array", target="coyote")
