"""Unit tests for the genetic algorithm."""
from unittest.mock import MagicMock

from core.search.algorithms.genetic_algorithm import GeneticAlgorithm
from core.search.evaluated_individual import EvaluatedIndividual
from core.search.fitness_value import FitnessValue
from core.search.individual import Individual
from core.search.service.randomness import Randomness


def build_genetic_algorithm(fitness_values):
    algorithm = GeneticAlgorithm(ff=MagicMock(), randomness=Randomness({"seed": 1}), stc=MagicMock(),
                                 archive=MagicMock(), config={}, mutator=MagicMock(), crossover=MagicMock(),
                                 sampler=MagicMock(), apc=MagicMock())
    algorithm.population = [EvaluatedIndividual(Individual(), FitnessValue(value, [])) for value in fitness_values]
    algorithm.population_size = len(fitness_values)
    return algorithm


def test_selection_never_returns_the_parent_itself():
    # The parent has the best fitness, so roulette selection would favour it the most
    algorithm = build_genetic_algorithm([0.1, 0.5, 0.9])

    selected = {algorithm.selection(0).fitness.value for _ in range(200)}

    assert selected == {0.5, 0.9}


def test_selection_returns_a_copy():
    algorithm = build_genetic_algorithm([0.1, 0.5])

    selected = algorithm.selection(0)

    assert selected is not algorithm.population[1]
    assert selected.fitness.value == 0.5


def test_selection_with_single_individual_returns_a_copy_of_it():
    algorithm = build_genetic_algorithm([0.3])

    assert algorithm.selection(0).fitness.value == 0.3
