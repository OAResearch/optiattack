from core.config_parser import ConfigParser
from core.search.algorithms.search_algorithm import SearchAlgorithm


class MioAlgorithm(SearchAlgorithm):

    """MIO search algorithm implementation."""
    def get_type(self):
        """Return the type of the search algorithm."""
        return ConfigParser.Algorithms.MIO

    def search_once(self):
        """Search for a solution."""

        if self.archive.is_empty() or self.randomness.next_bool(self.apc.get_probability_random_sampling()):
            individual = self.sampler.sample()
            ei = self.ff.calculate_fitness(individual)
            self.archive.add_archive_if_needed(ei)
            return

        sample = self.archive.sample_individual()

        if sample is not None:
            mutated = self.mutator.mutate(sample.individual)
            ei = self.ff.calculate_fitness(mutated)
            self.archive.add_archive_if_needed(ei, sample)
