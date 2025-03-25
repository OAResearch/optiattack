"""Random search algorithm implementation."""
from core.config_parser import ConfigParser
from core.search.algorithms.search_algorithm import SearchAlgorithm


class RandomAlgorithm(SearchAlgorithm):

    """Random search algorithm implementation."""

    def get_type(self):
        """Return the type of the search algorithm."""
        return ConfigParser.Algorithms.RANDOM_SEARCH

    def search_once(self):
        """Search for a solution."""

        individual = self.sampler.sample()
        ei = self.ff.calculate_fitness(individual)
        self.archive.add_archive_if_needed(ei)
