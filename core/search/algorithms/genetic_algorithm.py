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
            if not self.stc.should_continue_search():
                return
            individual = self.sampler.sample()
            ei = self.ff.calculate_fitness(individual)
            self.archive.add_archive_if_needed(ei)
            self.population.append(ei)

    def get_type(self):
        """Return the type of the search algorithm."""
        return ConfigParser.Algorithms.GENETIC

    def selection(self, parent_index: int):
        """Roulette wheel selection of a crossover partner other than the parent at parent_index."""
        if len(self.population) == 1:
            return self.population[0].copy()

        all_fitnesses = np.array([ind.fitness.value for ind in self.population])
        weights = np.exp(1.0 - all_fitnesses)
        weights[parent_index] = 0.0
        selection_probs = weights / np.sum(weights)
        selected = self.randomness.get_random_element(self.population, selection_probs=selection_probs)
        return selected.copy()

    def search_once(self):
        """Search for a solution."""

        for i in range(self.population_size):
            # Check before every evaluation so that the budget is never exceeded
            if not self.stc.should_continue_search():
                return
            parent1 = self.population[i].copy()
            parent2 = self.selection(i)

            self.crossover.apply_crossover(parent1.individual, parent2.individual)
            child1 = self.mutator.mutate(parent1.individual)
            child2 = self.mutator.mutate(parent2.individual)

            fitness1 = self.ff.calculate_fitness(child1)
            if self.stc.is_solution(fitness1.fitness) or not self.stc.should_continue_search():
                self.archive.add_archive_if_needed(fitness1)
                self.population[i] = fitness1
                return
            fitness2 = self.ff.calculate_fitness(child2)

            final_fitness = fitness1 if fitness1.fitness.value < fitness2.fitness.value else fitness2
            self.archive.add_archive_if_needed(final_fitness)
            self.population[i] = final_fitness
