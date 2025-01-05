import pytest
from unittest.mock import MagicMock
import sys
import io

from core.search.service.monitor.search_status_updater import SearchStatusUpdater


class TestSearchStatusUpdater:
    @pytest.fixture
    def updater(self):
        # Mock dependencies
        mock_time = MagicMock()
        mock_config = MagicMock()
        mock_archive = MagicMock()

        # Set up default mock behaviors
        mock_config.show_progress = True
        mock_time.percentage_used_budget.return_value = 0.5  # 50%
        mock_time.compute_executed_individual_time_statistics.return_value = (100.0, 10.0)  # avg_time, avg_size
        mock_time.get_seconds_since_last_improvement.return_value = 5  # 5 seconds
        mock_archive.number_of_covered_targets.return_value = 42  # 42 covered targets

        # Create the SearchStatusUpdater instance
        return SearchStatusUpdater(mock_time, mock_config, mock_archive)

    def test_new_action_evaluated_updates_status(self, updater):
        # Redirect stdout to capture printed output
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Call the method under test
        updater.new_action_evaluated()

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Verify the output
        output = captured_output.getvalue()
        assert "* Consumed search budget: 50.000%" in output
        assert "* Covered targets: 42; time per test: 100.0ms (10.0 actions); since last improvement: 5s" in output

    def test_new_action_evaluated_skips_when_disabled(self, updater):
        # Disable the updater
        updater.enabled = False

        # Redirect stdout to capture printed output
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Call the method under test
        updater.new_action_evaluated()

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Verify that nothing was printed
        assert captured_output.getvalue() == ""

    def test_up_line_and_erase(self, updater):
        # Redirect stdout to capture printed output
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Call the method under test
        updater.up_line_and_erase()

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Verify the output contains the ANSI escape sequences
        output = captured_output.getvalue()
        assert "\u001b[1A" in output  # Move up
        assert "\u001b[2K" in output  # Erase line