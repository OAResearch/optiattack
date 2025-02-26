"""Random search algorithm implementation."""

from core.search.algorithms.search_algorithm import SearchAlgorithm


class RandomAlgorithm(SearchAlgorithm):

    """Random search algorithm implementation."""

    def search_once(self):
        """Search for a solution."""

        individual = self.sampler.sample()
        ei = self.ff.calculate_fitness(individual)
        self.archive.add_archive_if_needed(ei)
