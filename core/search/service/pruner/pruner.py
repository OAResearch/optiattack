"""Abstract class for pruners."""

from core.search.service.archive import Archive
from core.search.service.fitness_function.fitness_function import FitnessFunction
from core.search.service.monitor.search_status_updater import SearchStatusUpdater


class Pruner:

    """Abstract class for pruners."""

    def __init__(self, archive: Archive, ff: FitnessFunction, ssu: SearchStatusUpdater):
        """Initialize the pruner."""
        self.archive = archive
        self.ff = ff
        self.ssu = ssu

    def pruner_type(self):
        """Return the type of the pruner."""
        pass

    def minimize_actions_in_archive(self):
        """Minimize the actions in the archive."""
        pass
