"""Class used to keep track of passing of time during the search."""

import logging
import time

from core.config_parser import ConfigParser


class SearchTimeController:

    """Class used to keep track of passing of time during the search."""

    def __init__(self, config):
        """Initialize the search time controller."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.evaluated_individuals = 0
        self.last_action_improvement_timestamp = 0
        self.start_time = 0
        self.search_started = False

    def start_search(self):
        """Start the search time controller"""
        self.start_time = time.time()
        self.search_started = True
        self.last_action_improvement_timestamp = self.start_time

    def new_individual_evaluation(self):
        """Update the number of evaluated individuals."""
        self.evaluated_individuals += 1

    def get_elapsed_seconds(self):
        """Get the elapsed time in seconds."""
        return time.time() - self.start_time

    def get_elapsed_time(self):
        """Get the elapsed time in HH:MM:SS format."""
        return time.strftime('%H:%M:%S', time.gmtime(self.get_elapsed_seconds()))

    def percentage_used_budget(self):
        """Get the percentage of the budget used."""
        if not self.search_started:
            return 0.0

        if self.config.get('stopping_criterion') == ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS:
            return self.evaluated_individuals / self.config.get('max_evaluations')

        elif self.config.get('stopping_criterion') == ConfigParser.StoppingCriterion.TIME:
            return self.get_elapsed_seconds() / self.config.get('max_evaluations')

        else:
            raise ValueError("Not supported stopping criterion")

    def should_continue_search(self):
        """Check if the search should continue."""
        return self.percentage_used_budget() < 1.0
