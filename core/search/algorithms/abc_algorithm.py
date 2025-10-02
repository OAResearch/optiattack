"""Artificial Bee Colony algorithm implementation."""
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


class ArtificialBeeColonyAlgorithm(SearchAlgorithm):

    """Artificial Bee Colony algorithm implementation."""

    def __init__(self, ff: FitnessFunction,
                 randomness: Randomness,
                 stc: SearchTimeController,
                 archive: Archive,
                 config: dict,
                 mutator: Mutator,
                 crossover: Crossover,
                 sampler: Sampler,
                 apc: AdaptiveParameterControl):
        """Initialize the ABC algorithm."""
        super().__init__(ff, randomness, stc, archive, config, mutator, crossover, sampler, apc)
        self.food_sources = list[EvaluatedIndividual]()
        self.colony_size = None
        self.limit = None  # Abandon limit
        self.trial_counters = None

    def setup_before_search(self):
        """Setup the ABC algorithm before the search starts."""
        self.colony_size = self.config.get("population_size")
        self.limit = self.config.get("abc_limit", self.colony_size * 10)
        self.food_sources: list[EvaluatedIndividual] = []
        self.trial_counters = []

        # Initialize food sources (employed bees phase)
        for _ in range(self.colony_size):
            individual = self.sampler.sample()
            ei = self.ff.calculate_fitness(individual)
            self.archive.add_archive_if_needed(ei)
            self.food_sources.append(ei)
            self.trial_counters.append(0)

    def get_type(self):
        """Return the type of the search algorithm."""
        return ConfigParser.Algorithms.ABC

    def employed_bee_phase(self):
        """Employed bees explore food sources."""
        for i in range(self.colony_size):
            # Select a random food source different from current one
            k = self.randomness.next_int(0, self.colony_size - 1)
            while k == i:
                k = self.randomness.next_int(0, self.colony_size - 1)

            # Create new candidate solution
            current = self.food_sources[i].copy()
            neighbor = self.food_sources[k].copy()

            # Apply crossover between current and neighbor, then mutate
            self.crossover.apply_crossover(current.individual, neighbor.individual)
            new_individual = self.mutator.mutate(current.individual)
            new_ei = self.ff.calculate_fitness(new_individual)

            # Greedy selection
            if new_ei.fitness.value < self.food_sources[i].fitness.value:
                self.food_sources[i] = new_ei
                self.trial_counters[i] = 0
                self.archive.add_archive_if_needed(new_ei)
            else:
                self.trial_counters[i] += 1

    def calculate_probabilities(self):
        """Calculate selection probabilities for onlooker bees."""
        all_fitnesses = np.array([fs.fitness.value for fs in self.food_sources])
        # Convert to maximization (lower fitness is better, so we invert)
        scaled_fitness = 1.0 - all_fitnesses
        # Ensure non-negative values
        scaled_fitness = np.maximum(scaled_fitness, 0)
        total = np.sum(scaled_fitness)
        if total == 0:
            # If all fitnesses are the same, use uniform probability
            return np.ones(self.colony_size) / self.colony_size
        return scaled_fitness / total

    def onlooker_bee_phase(self):
        """Onlooker bees select food sources based on probability."""
        probabilities = self.calculate_probabilities()

        for _ in range(self.colony_size):
            # Roulette wheel selection
            i = self.randomness.get_random_element(
                list(range(self.colony_size)),
                selection_probs=probabilities
            )

            # Select a random food source different from selected one
            k = self.randomness.next_int(0, self.colony_size - 1)
            while k == i:
                k = self.randomness.next_int(0, self.colony_size - 1)

            # Create new candidate solution
            current = self.food_sources[i].copy()
            neighbor = self.food_sources[k].copy()

            # Apply crossover between current and neighbor, then mutate
            self.crossover.apply_crossover(current.individual, neighbor.individual)
            new_individual = self.mutator.mutate(current.individual)
            new_ei = self.ff.calculate_fitness(new_individual)

            # Greedy selection
            if new_ei.fitness.value < self.food_sources[i].fitness.value:
                self.food_sources[i] = new_ei
                self.trial_counters[i] = 0
                self.archive.add_archive_if_needed(new_ei)
            else:
                self.trial_counters[i] += 1

    def scout_bee_phase(self):
        """Scout bees abandon exhausted food sources and discover new ones."""
        for i in range(self.colony_size):
            if self.trial_counters[i] >= self.limit:
                # Abandon the food source and generate a new one
                new_individual = self.sampler.sample()
                new_ei = self.ff.calculate_fitness(new_individual)
                self.food_sources[i] = new_ei
                self.trial_counters[i] = 0
                self.archive.add_archive_if_needed(new_ei)

    def search_once(self):
        """Execute one iteration of ABC algorithm."""
        self.employed_bee_phase()
        self.onlooker_bee_phase()
        self.scout_bee_phase()
