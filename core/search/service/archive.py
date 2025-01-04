"""Module contains the Archive class, which is a base class for the archives used in the search service."""


class Archive:

    """Base class for the archives used in the search service."""

    def __init__(self):
        """Initialize the archive."""

        self.populations = []
        self.sampling_counter = 0
        self.last_improvement = []
        self.last_chosen = []

    def clean_population(self):
        """Clean the populations list."""
        self.populations = []

    def add_archive_if_needed(self, individual):
        """Add an individual to the archive if it is better than the current best solution."""
        raise NotImplementedError("This method should be implemented")

    def shrink_archive(self):
        """Remove individuals from the archive that do not improve the fitness of the current best solution."""
        raise NotImplementedError("This method should be implemented")

    def sample_individual(self):
        """Sample an individual from the archive."""
        raise NotImplementedError("This method should be implemented")

    def is_empty(self):
        """Check if the archive is empty."""
        return len(self.populations) == 0
