"""Standard pruner that tries to prune actions by removing each action at a time."""

from core.search.service.pruner.pruner import Pruner


class StandardPruner(Pruner):

    """Standard pruner that try to prune actions removing each actions at a time."""

    def pruner_type(self):
        """Return the type of the pruner."""
        return "standard"

    def minimize_actions_in_archive(self):
        """Minimize the actions in the archive."""
        # TODO minimize the actions in the archive
        return False
