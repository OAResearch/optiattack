"""Module contains the Archive class, which is a base class for the archives used in the search service."""


class Archive:

    """Base class for the archives used in the search service."""

    def __init__(self, search_time_controller, randomness):
        """Initialize the archive."""

        self.populations = []
        self.sampling_counter = 0
        self.last_improvement = []
        self.last_chosen = []
        self.search_time_controller = search_time_controller
        self.randomness = randomness

    def clean_population(self):
        """Clean the populations list."""
        self.populations = []

    def add_archive_if_needed(self, individual):
        """Add an individual to the archive if it is better than the current best solution."""
        if self.randomness.next_bool(0.5):
            self.populations.append(individual)
            self.search_time_controller.new_action_improvement()

    def shrink_archive(self):
        """Remove individuals from the archive that do not improve the fitness of the current best solution."""
        raise NotImplementedError("This method should be implemented")

    def sample_individual(self):
        """Sample an individual from the archive."""
        raise NotImplementedError("This method should be implemented")

    def is_empty(self):
        """Check if the archive is empty."""
        return len(self.populations) == 0

    def number_of_population(self):
        """Return the number of population in the archive."""
        return len(self.populations)
