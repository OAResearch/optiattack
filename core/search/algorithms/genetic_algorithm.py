"""Genetic algorithm implementation."""
import numpy as np

from core.config_parser import ConfigParser
from core.search.algorithms.search_algorithm import SearchAlgorithm
from core.search.evaluated_individual import EvaluatedIndividual
from core.search.service.adaptive_parameter_control import AdaptiveParameterControl
from core.search.service.archive import Archive
from core.search.service.crossover.crossover import Crossover
from core.search.service.fitness_function.fitness_function import FitnessFunction
from core.search.service.mutator.mutator import Mutator
from core.search.service.randomness import Randomness
from core.search.service.sampler.sampler import Sampler
from core.search.service.search_time_controller import SearchTimeController


class GeneticAlgorithm(SearchAlgorithm):

    """Genetic algorithm implementation."""

    def __init__(self, ff: FitnessFunction,
                 randomness: Randomness,
                 stc: SearchTimeController,
                 archive: Archive,
                 config: dict,
                 mutator: Mutator,
                 crossover: Crossover,
                 sampler: Sampler,
                 apc: AdaptiveParameterControl):
        """Initialize the genetic algorithm."""
        super().__init__(ff, randomness, stc, archive, config, mutator, crossover, sampler, apc)
        self.population = list[EvaluatedIndividual]()
        self.population_size = None

    def setup_before_search(self):
        """Setup the genetic algorithm before the search starts."""
        self.population_size = self.config.get("population_size")
        self.population: list[EvaluatedIndividual] = []

        for _ in range(self.population_size):
            individual = self.sampler.sample()
            ei = self.ff.calculate_fitness(individual)
            self.archive.add_archive_if_needed(ei)
            self.population.append(ei)

    def get_type(self):
        """Return the type of the search algorithm."""
        return ConfigParser.Algorithms.GENETIC

    def selection(self, parent1: EvaluatedIndividual):
        """Roullette wheel selection to select a parent for crossover."""
        all_fitnesses = np.array([ind.fitness.value for ind in self.population])
        scaled_fitness = 1.0 - all_fitnesses
        selection_probs = np.exp(scaled_fitness) / np.sum(np.exp(scaled_fitness))
        selected = self.randomness.get_random_element(self.population, selection_probs=selection_probs)
        while parent1 == selected:
            selected = self.randomness.get_random_element(self.population, selection_probs=selection_probs)
        return selected.copy()

    def search_once(self):
        """Search for a solution."""

        for i in range(self.population_size):
            parent1 = self.population[i].copy()
            parent2 = self.selection(parent1)

            self.crossover.apply_crossover(parent1.individual, parent2.individual)
            child1 = self.mutator.mutate(parent1.individual)
            child2 = self.mutator.mutate(parent2.individual)

            fitness1 = self.ff.calculate_fitness(child1)
            fitness2 = self.ff.calculate_fitness(child2)

            final_fitness = fitness1 if fitness1.fitness.value < fitness2.fitness.value else fitness2
            self.archive.add_archive_if_needed(final_fitness)
            self.population[i] = final_fitness
