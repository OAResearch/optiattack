import pytest
from unittest.mock import MagicMock, patch
from core.search.solution import Solution
from core.search.service.pruner.standard_pruner import StandardPruner


@pytest.fixture
def standard_pruner():
    # Create mock dependencies
    archive = MagicMock()
    ff = MagicMock()
    ssu = MagicMock()

    # Initialize StandardPruner with mock dependencies
    pruner = StandardPruner(archive, ff, ssu)
    return pruner


def test_pruner_type(standard_pruner):
    assert standard_pruner.pruner_type() == "standard"


def test_minimize_actions_in_archive_successful_removal(standard_pruner):
    # Setup test data
    original_actions = [MagicMock(), MagicMock(), MagicMock()]
    solution = Solution(original_actions.copy(), MagicMock(value=1.0))
    standard_pruner.archive.extract_solution.return_value = solution

    # Mock fitness calculation to return 0 when second action is removed
    def mock_calculate_fitness(actions):
        if len(actions) == 2 and actions[0] == original_actions[0] and actions[1] == original_actions[2]:
            return MagicMock(value=0.0)
        return MagicMock(value=1.0)

    standard_pruner.ff.calculate_fitness_with_actions.side_effect = mock_calculate_fitness

    # Execute
    result = standard_pruner.minimize_actions_in_archive()

    # Verify
    assert len(result.actions) == 2
    assert original_actions[1] not in result.actions  # Second action should be removed
    standard_pruner.ssu.start_minimization.assert_called_once()
    standard_pruner.ssu.remove_action.assert_called_once_with(2, 3, original_actions[1])
    standard_pruner.ssu.kept_action.assert_any_call(1, 3, original_actions[0])
    standard_pruner.ssu.kept_action.assert_any_call(3, 3, original_actions[2])


def test_minimize_actions_in_archive_no_removal(standard_pruner):
    # Setup test data
    original_actions = [MagicMock(), MagicMock()]
    solution = Solution(original_actions.copy(), MagicMock(value=1.0))
    standard_pruner.archive.extract_solution.return_value = solution

    # Mock fitness calculation to never return 0
    standard_pruner.ff.calculate_fitness_with_actions.return_value = MagicMock(value=1.0)

    # Execute
    result = standard_pruner.minimize_actions_in_archive()

    # Verify
    assert len(result.actions) == 2
    assert result.actions == original_actions
    standard_pruner.ssu.start_minimization.assert_called_once()
    standard_pruner.ssu.remove_action.assert_not_called()
    standard_pruner.ssu.kept_action.assert_any_call(1, 2, original_actions[0])
    standard_pruner.ssu.kept_action.assert_any_call(2, 2, original_actions[1])


def test_minimize_actions_in_archive_all_removed(standard_pruner):
    # Setup test data
    original_actions = [MagicMock(), MagicMock()]
    solution = Solution(original_actions.copy(), MagicMock(value=1.0))
    standard_pruner.archive.extract_solution.return_value = solution

    # Mock fitness calculation to always return 0 when any action is removed
    standard_pruner.ff.calculate_fitness_with_actions.return_value = MagicMock(value=0.0)

    # Execute
    result = standard_pruner.minimize_actions_in_archive()

    # Verify
    assert len(result.actions) == 0
    standard_pruner.ssu.start_minimization.assert_called_once()
    assert standard_pruner.ssu.remove_action.call_count == 2
    standard_pruner.ssu.kept_action.assert_not_called()


def test_minimize_actions_in_archive_with_empty_solution(standard_pruner):
    # Setup empty solution
    solution = Solution([], MagicMock(value=0.0))
    standard_pruner.archive.extract_solution.return_value = solution

    # Execute
    result = standard_pruner.minimize_actions_in_archive()

    # Verify
    assert len(result.actions) == 0
    standard_pruner.ssu.start_minimization.assert_called_once()
    standard_pruner.ssu.remove_action.assert_not_called()
    standard_pruner.ssu.kept_action.assert_not_called()


def test_minimize_actions_in_archive_counter_adjustment(standard_pruner):
    # Setup test data with 3 actions
    original_actions = [MagicMock(), MagicMock(), MagicMock()]
    solution = Solution(original_actions.copy(), MagicMock(value=1.0))
    standard_pruner.archive.extract_solution.return_value = solution

    # Mock fitness calculation to return 0 when middle action is removed
    def mock_calculate_fitness(actions):
        if len(actions) == 2 and actions[0] == original_actions[0] and actions[1] == original_actions[2]:
            return MagicMock(value=0.0)
        return MagicMock(value=1.0)

    standard_pruner.ff.calculate_fitness_with_actions.side_effect = mock_calculate_fitness

    # Execute
    result = standard_pruner.minimize_actions_in_archive()

    # Verify counter adjustment when action is removed
    assert len(result.actions) == 2
    # Should process all 3 original actions (counter goes 0,1,2)
    # When counter=1 (second action) is removed, counter decrements to 0
    # Then increments to 1 to process third action
    standard_pruner.ssu.remove_action.assert_called_once_with(2, 3, original_actions[1])
    assert standard_pruner.ssu.kept_action.call_count == 2