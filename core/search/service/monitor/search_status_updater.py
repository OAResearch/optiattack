"""Module is responsible for updating the search status on the console."""
import sys
import time

from core.config_parser import ConfigParser
from core.search.service.archive import Archive
from core.search.service.monitor.search_listener import SearchListener
from core.search.service.search_time_controller import SearchTimeController


class SearchStatusUpdater(SearchListener):

    """Class responsible for updating the search status on the console."""

    def __init__(self, search_time_controller: SearchTimeController, config, archive: Archive):
        """Initialize the search status updater."""
        self.search_time_controller = search_time_controller
        self.config = config
        self.archive = archive
        self.passed = "-1"
        self.last_update_ms = 0
        self.last_coverage_computation = 0
        self.coverage = 0
        self.utf8 = 'utf-8'
        self.first = True
        self.out = sys.stdout

        if self.config.get("show_progress") is True:
            self.search_time_controller.add_listener(self)

    @staticmethod
    def erase_line():
        """Erase the current line on the console."""
        print("\u001b[2K", end='')  # erase line

    @staticmethod
    def move_up():
        """Move up by one line on the console."""
        print("\u001b[1A", end='')  # move up by 1 line

    @staticmethod
    def up_line_and_erase():
        """Move up by one line and erase the line on the console."""
        SearchStatusUpdater.move_up()
        SearchStatusUpdater.erase_line()

    def new_action_evaluated(self):
        """Update the search status on the console."""
        percentage_int = int(self.search_time_controller.percentage_used_budget() * 100)
        current = "{:.3f}".format(self.search_time_controller.percentage_used_budget() * 100)

        if self.first:
            print()
            print()
            self.first = False

        delta = int(time.time() * 1000) - self.last_update_ms

        # Writing on console is I/O, which is expensive. So, can't do it too often
        if current != self.passed and delta > 500:
            self.last_update_ms += delta
            self.passed = current

            if percentage_int - self.last_coverage_computation > 0:
                self.last_coverage_computation = percentage_int
                self.coverage = self.archive.number_of_population()

            avg_time_and_size = self.search_time_controller.compute_executed_individual_time_statistics()
            avg_time = "{:.1f}".format(avg_time_and_size[0])
            avg_size = "{:.1f}".format(avg_time_and_size[1])

            since_last = self.search_time_controller.get_seconds_since_last_improvement()

            self.up_line_and_erase()
            self.up_line_and_erase()
            print(f"* Consumed search budget: {current}%")
            print(f"* Archive size: {self.coverage}; "
                  f"time per test: {avg_time}ms ({avg_size} actions); "
                  f"since last improvement: {since_last}s")
