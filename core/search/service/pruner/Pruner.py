from logging import Logger

from core.search.action import Action
from core.search.service.archive import Archive


class Pruner:

    """Abstract class for pruners."""

    def __init__(self, logger: Logger, archive: Archive):
        """Initialize the pruner."""
        self.archive = archive
        self.logger = logger

    def pruner_type(self):
        """Return the type of the pruner."""
        pass

    def minimize_actions_in_archive(self):
        """Return the type of the pruner."""
        pass

    def print_progress(self, action:Action):
        self.logger.info(f"Action {action} is trying to be pruned")
