"""Interface class for search listeners."""


class SearchListener:

    """Interface class for search listeners."""

    def new_individual_evaluated(self):
        """Notify that a new individual has been evaluated."""
        pass
