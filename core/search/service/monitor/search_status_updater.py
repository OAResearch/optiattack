"""Module is responsible for updating the search status on the console."""
import sys
import time

from core.search.service.archive import Archive
from core.search.service.monitor.search_listener import SearchListener
from core.search.service.search_time_controller import SearchTimeController


class SearchStatusUpdater(SearchListener):

    """Class responsible for updating the search status on the console."""

    def __init__(self, stc: SearchTimeController, config, archive: Archive):
        """Initialize the search status updater."""
        self.stc = stc
        self.config = config
        self.archive = archive
        self.passed = "-1"
        self.last_update_ms = 0
        self.last_coverage_computation = 0
        self.action_size = 0
        self.utf8 = 'utf-8'
        self.first = True
        self.out = sys.stdout
        self.enabled = self.config.get("show_progress")

        if self.enabled is True:
            self.stc.add_listener(self)

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
        if not self.enabled:
            return

        percentage_int = int(self.stc.percentage_used_budget() * 100)
        current = "{:.3f}".format(self.stc.percentage_used_budget() * 100)

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
                self.action_size = self.archive.get_actions().__len__()

            avg_time_and_size = self.stc.compute_executed_individual_time_statistics()
            avg_time = "{:.1f}".format(avg_time_and_size[0])
            avg_size = "{:.1f}".format(avg_time_and_size[1])

            since_last = self.stc.get_seconds_since_last_improvement()

            self.up_line_and_erase()
            self.up_line_and_erase()
            print(f"* Consumed search budget: {current}%")
            print(f"* Action size: {self.action_size}; "
                  f"time per test: {avg_time}ms ({avg_size} actions); "
                  f"since last improvement: {since_last}s; "
                  f"fitness: {self.stc.get_current_fitness_value()}")

    class bcolors:

        """Class for color codes."""

        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    def search_end(self):
        """Update the search status on the console."""
        self.up_line_and_erase()
        self.up_line_and_erase()

        print(f"{self.bcolors.OKGREEN}Number of evaluated individuals: {self.stc.get_evaluated_individuals()}")
        print(f"Number of actions: {self.archive.get_actions().__len__()}")
        print(f"Best fitness: {self.stc.get_current_fitness_value()}")
        print(f"Elapsed time: {self.stc.get_elapsed_time()}")

        self.up_line_and_erase()
        print("Predictions")
        print("-----------------")
        prediction_beginning = self.archive.get_original_prediction_results().predictions[0]
        prediction_end = self.stc.current_fitness_value.predictions[0]
        print(f"{self.bcolors.ENDC}Beginning prediction: {self.bcolors.OKBLUE}{prediction_beginning.label} "
              f"{self.bcolors.ENDC}with confidence level: {self.bcolors.WARNING}{prediction_beginning.value}")
        self.up_line_and_erase()
        print(f"{self.bcolors.ENDC}End prediction: {self.bcolors.OKBLUE}{prediction_end.label} "
              f"{self.bcolors.ENDC}with confidence level: {self.bcolors.OKGREEN}{prediction_end.value}")

        self.up_line_and_erase()
        self.up_line_and_erase()

        print(f"Search ended.{self.bcolors.ENDC}")
