"""Phase controller for managing the different phases of the search process."""


class PhaseController:

    """Class is responsible for managing the different phases of the search process."""

    class Phase:

        """Phases of the search process."""

        NOT_STARTED = "not_started"
        SEARCH = "search"
        PRUNING = "pruning"
        END = "end"

    def __init__(self):
        """Initialize the Phases class with a search object."""
        self.current_phase = self.Phase.NOT_STARTED

    def start(self):
        """Initialize the search phase."""
        self.current_phase = self.Phase.SEARCH

    def run_search(self):
        """Run the search phase."""
        self.current_phase = self.Phase.SEARCH

    def prune(self):
        """Run the pruning phase."""
        if self.current_phase == self.Phase.NOT_STARTED or self.current_phase == self.Phase.END:
            raise ValueError("Cannot run minimization phase before search phase.")

        self.current_phase = self.Phase.PRUNING

    def end(self):
        """Finalize the search phase."""
        self.current_phase = self.Phase.END

    def is_pruning(self):
        """Check if the current phase is pruning."""
        return self.current_phase == self.Phase.PRUNING

    def is_search(self):
        """Check if the current phase is search."""
        return self.current_phase == self.Phase.SEARCH

    def is_not_started(self):
        """Check if the current phase is not started."""
        return self.current_phase == self.Phase.NOT_STARTED

    def is_end(self):
        """Check if the current phase is end."""
        return self.current_phase == self.Phase.END
