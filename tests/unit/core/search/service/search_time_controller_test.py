import time
from time import sleep
from unittest.mock import MagicMock

import pytest

from core.config_parser import ConfigParser
from core.search.service.search_time_controller import SearchTimeController
from main import OptiAttack
from core.problem.base_module import BaseModule


@pytest.fixture
def container():
    container = BaseModule()
    yield container
    container.unwire()


def get_app(container, config_setting=None):
    container.config.override(config_setting)
    app = OptiAttack(
        config=container.config(),
        randomness=container.randomness(),
        stc=container.stc(),
    )
    assert app.config.get("stopping_criterion") == config_setting.get("stopping_criterion")
    assert app.config.get("max_evaluations") == config_setting.get("max_evaluations")
    return app


def test_individual_evaluations(container):
    app = get_app(container,
                  {"stopping_criterion": ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS,
                   "max_evaluations": 100,
                   "seed": 42})

    app.stc.start_search()
    for i in range(100):
        app.stc.new_individual_evaluation()

    assert app.stc.should_continue_search() is False


def test_time(container):
    app = get_app(container,
                  {"stopping_criterion": ConfigParser.StoppingCriterion.TIME,
                   "max_evaluations": 10,
                   "seed": 42})

    app.stc.start_search()
    sleep(10)
    assert app.stc.percentage_used_budget() >= 1.0
    assert app.stc.should_continue_search() is False

@pytest.fixture
def search_time_controller():
    """Fixture to create a SearchTimeController instance with a mock config."""
    mock_config = MagicMock()
    mock_config.get.return_value = 100  # Default value for max_evaluations
    mock_config.get.side_effect = lambda key: {
        'max_evaluations': 100,
        'stopping_criterion': ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS
    }.get(key)
    return SearchTimeController(mock_config)

def test_start_search(search_time_controller):
    """Test that the search start time is set correctly."""
    search_time_controller.start_search()
    assert search_time_controller.search_started
    assert search_time_controller.start_time > 0
    assert search_time_controller.last_action_improvement_timestamp == search_time_controller.start_time

def test_new_individual_evaluation(search_time_controller):
    """Test that the number of evaluated individuals is incremented and listeners are notified."""
    mock_listener = MagicMock()
    search_time_controller.add_listener(mock_listener)

    search_time_controller.new_individual_evaluation()
    assert search_time_controller.evaluated_individuals == 1
    mock_listener.new_action_evaluated.assert_called_once()

def test_new_action_improvement(search_time_controller):
    """Test that the last action improvement timestamp is updated."""
    search_time_controller.start_search()
    initial_timestamp = search_time_controller.last_action_improvement_timestamp

    time.sleep(0.1)  # Simulate a delay
    search_time_controller.new_action_improvement()

    assert search_time_controller.last_action_improvement_timestamp > initial_timestamp

def test_get_elapsed_seconds(search_time_controller):
    """Test that the elapsed time in seconds is calculated correctly."""
    search_time_controller.start_search()
    time.sleep(0.1)  # Simulate a delay
    elapsed_seconds = search_time_controller.get_elapsed_seconds()
    assert elapsed_seconds >= 0.1

def test_get_elapsed_time(search_time_controller):
    """Test that the elapsed time is formatted correctly."""
    search_time_controller.start_search()
    time.sleep(1)  # Simulate a delay
    elapsed_time = search_time_controller.get_elapsed_time()
    assert elapsed_time == "00:00:01"

def test_percentage_used_budget_evaluations(search_time_controller):
    """Test the percentage of the budget used for individual evaluations."""
    search_time_controller.start_search()
    search_time_controller.new_individual_evaluation()
    assert search_time_controller.percentage_used_budget() == 0.01  # 1 / 100

def test_percentage_used_budget_time(search_time_controller):
    """Test the percentage of the budget used for time-based stopping criterion."""
    mock_config = MagicMock()
    mock_config.get.return_value = 10  # 10 seconds max
    mock_config.get.side_effect = lambda key: {
        'max_evaluations': 10,
        'stopping_criterion': ConfigParser.StoppingCriterion.TIME
    }.get(key)
    search_time_controller.config = mock_config

    search_time_controller.start_search()
    time.sleep(1)  # Simulate 1 second delay
    assert search_time_controller.percentage_used_budget() == pytest.approx(0.1, rel=1e-2)  # 1 / 10

def test_should_continue_search(search_time_controller):
    """Test that the search should continue if the budget is not exhausted."""
    search_time_controller.start_search()
    assert search_time_controller.should_continue_search()

    # Exhaust the budget
    for _ in range(100):
        search_time_controller.new_individual_evaluation()
    assert not search_time_controller.should_continue_search()

def test_report_executed_individual_time(search_time_controller):
    """Test that executed individual times are reported correctly."""
    search_time_controller.report_executed_individual_time(100, 10)
    search_time_controller.report_executed_individual_time(200, 20)

    assert len(search_time_controller.executed_individual_time) == 2
    assert search_time_controller.executed_individual_time[0] == (100, 10)
    assert search_time_controller.executed_individual_time[1] == (200, 20)

def test_compute_executed_individual_time_statistics(search_time_controller):
    """Test that statistics are computed correctly for executed individual times."""
    search_time_controller.report_executed_individual_time(100, 10)
    search_time_controller.report_executed_individual_time(200, 20)

    avg_ms, avg_actions = search_time_controller.compute_executed_individual_time_statistics()
    assert avg_ms == 150.0
    assert avg_actions == 15.0

def test_get_seconds_since_last_improvement(search_time_controller):
    """Test that the seconds since the last improvement are calculated correctly."""
    search_time_controller.start_search()
    search_time_controller.new_action_improvement()
    time.sleep(1)  # Simulate a delay
    seconds_since_improvement = search_time_controller.get_seconds_since_last_improvement()
    assert seconds_since_improvement == pytest.approx(1, rel=1e-2)