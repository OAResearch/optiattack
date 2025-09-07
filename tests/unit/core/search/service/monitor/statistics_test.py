import pytest
import os
import json
import numpy as np
from unittest.mock import MagicMock


from core.search.evaluated_individual import EvaluatedIndividual
from core.search.fitness_value import FitnessValue
from core.search.service.archive import Archive
from core.search.service.monitor.statistics import Statistics
from core.search.service.search_time_controller import SearchTimeController
from core.search.solution import Solution

# Fixture for the Statistics instance
@pytest.fixture
def statistics():
    stc = MagicMock(spec=SearchTimeController)
    archive = MagicMock(spec=Archive)
    archive.populations = []
    config = {
        "snapshot_interval": 10,
        "seed": 123,
        "experiment_label": "test_experiment",
        "output_dir": "test_output",
        "write_statistics": True,
        "save_images": True,
        "show_plots": False,
        "image_width": 100,
        "image_height": 100
    }
    return Statistics(stc, archive, config)


def test_new_action_evaluated_with_snapshot(statistics):
    # Set the snapshot threshold to trigger a snapshot
    statistics.snapshot_threshold = 10
    statistics.stc.percentage_used_budget.return_value = 0.1  # 10%

    # Mock the take_snapshot method
    statistics.take_snapshot = MagicMock()

    # Call the new_action_evaluated method
    statistics.new_action_evaluated()

    # Assert that take_snapshot was called
    statistics.take_snapshot.assert_called_once()

def test_new_action_evaluated_without_snapshot(statistics):
    # Set the snapshot threshold to not trigger a snapshot
    statistics.snapshot_threshold = 20
    statistics.stc.percentage_used_budget.return_value = 0.1  # 10%

    # Mock the take_snapshot method
    statistics.take_snapshot = MagicMock()

    # Call the new_action_evaluated method
    statistics.new_action_evaluated()

    # Assert that take_snapshot was not called
    statistics.take_snapshot.assert_not_called()

def test_take_snapshot(statistics):
    # Mock the extract_solution method of the archive
    solution = MagicMock(spec=Solution)
    statistics.archive.extract_solution.return_value = solution

    # Call the take_snapshot method
    statistics.take_snapshot()

    # Assert that the solution was added to the snapshots
    assert len(statistics.snapshots) == 1
    assert statistics.snapshots[0] == solution
    assert statistics.snapshot_threshold == 20  # 10 (initial) + 10 (interval)

def test_get_data(statistics):
    # Mock a solution
    solution = MagicMock(spec=Solution)
    solution.fitness_value = MagicMock(spec=FitnessValue)
    solution.fitness_value.value = 0.5
    solution.fitness_value.predictions = [MagicMock(label="label1", value=0.1)]
    solution.actions = [MagicMock(spec=EvaluatedIndividual), MagicMock(spec=EvaluatedIndividual), MagicMock(spec=EvaluatedIndividual)]

    # solution.actions.__len__.return_value = 3

    # Mock the stc methods
    statistics.stc.get_evaluated_individuals.return_value = 100
    statistics.stc.percentage_used_budget.return_value = 0.5

    # Call the get_data method
    data = statistics.get_data(solution)

    # Assert that the data is correct
    assert data == {
        'eval_count': 100,
        'interval_count': 10,
        'current_fitness': 0.5,
        'predictions': solution.fitness_value.predictions,
        'archive_size': 3,
        'percentage_used_budget': 0.5,
        'archive': statistics.archive.populations.copy()
    }

def test_write_statistics(statistics):
    # Mock the methods called by write_statistics
    statistics.take_snapshot = MagicMock()
    statistics.write_snapshots = MagicMock()
    statistics.report_statistics = MagicMock()
    statistics.save_images = MagicMock()

    # Call the write_statistics method
    statistics.write_statistics()

    # Assert that the methods were called
    statistics.take_snapshot.assert_called_once()
    statistics.write_snapshots.assert_called_once()
    statistics.report_statistics.assert_called_once()
    statistics.save_images.assert_called_once()

def test_save_final_image(statistics):
    # Mock the get_mutated_image method of the archive
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    statistics.archive.get_mutated_image.return_value = img_array

    # Call the save_final_image method
    statistics.save_final_image()

    # Assert that the image was saved
    assert os.path.exists(f"{statistics.output_dir}/{statistics.final_image_name}.jpg")

def test_save_matrix_overlay(statistics):
    # Mock the extract_solution method of the archive
    solution = MagicMock(spec=Solution)
    action = MagicMock()
    action.get_location.return_value = (50, 50)
    action.get_color.return_value = [255, 0, 0]
    solution.actions = [action]
    statistics.archive.extract_solution.return_value = solution

    # Call the save_matrix_overlay method
    statistics.save_matrix_overlay()

    # Assert that the image was saved
    assert os.path.exists(f"{statistics.output_dir}/{statistics.overlay_image_name}.png")

def test_save_line_plot(statistics):
    # Mock the snapshots
    solution = MagicMock(spec=Solution)
    solution.fitness_value = MagicMock(spec=FitnessValue)
    prediction = MagicMock(label="label1", value=0.1)
    solution.fitness_value.predictions = [prediction]
    statistics.snapshots = [solution]

    # Call the save_line_plot method
    statistics.save_line_plot()

    # Assert that the plot was saved
    assert os.path.exists(f"{statistics.output_dir}/{statistics.line_plot_name}.png")

def test_save_statistics(statistics):
    # Mock the snapshots and stc methods
    solution = MagicMock(spec=Solution)
    solution.fitness_value = MagicMock(spec=FitnessValue)
    prediction = MagicMock(label="label1", value=0.1)
    solution.fitness_value.predictions = [prediction]
    solution.actions = [MagicMock(get_location=lambda: (50, 50), get_color=lambda: [255, 0, 0])]
    statistics.snapshots = [solution]
    statistics.stc.get_elapsed_seconds.return_value = 10
    statistics.stc.get_evaluated_individuals.return_value = 100
    statistics.stc.current_fitness_value = MagicMock(spec=FitnessValue)
    statistics.stc.current_fitness_value.value = 0.5

    # Call the save_data_as_json method
    statistics.save_statistics()

    # Assert that the JSON file was saved
    assert os.path.exists(f"{statistics.output_dir}/{statistics.statistics_file_name}.json")
    assert os.path.exists(f"{statistics.output_dir}/{statistics.statistics_file_name}.csv")

    # Load the JSON file and verify its contents
    with open(f"{statistics.output_dir}/{statistics.statistics_file_name}.json", "r") as file:
        data = json.load(file)
        assert data["execution_time"] == 10
        assert data["eval_count"] == 100
        assert data["current_fitness"] == 0.5
        # TODO - we need to fix and activate these assertions.
        # assert data["predictions"] == {"label1": [0.1]}
        # assert data["action_size"] == 1
        # assert data["changes"] == [{"location": (50, 50), "color": [255, 0, 0]}]
        assert data["config"] == statistics.config
