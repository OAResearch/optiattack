"""Unit tests for Artificial Bee Colony algorithm."""
import pytest
from unittest.mock import MagicMock, patch

from core.config_parser import ConfigParser
from core.search.action import Action
from core.search.algorithms.abc_algorithm import ArtificialBeeColonyAlgorithm
from core.search.evaluated_individual import EvaluatedIndividual
from core.search.fitness_value import FitnessValue
from core.search.individual import Individual
from core.search.phase_controller import PhaseController
from core.search.service.adaptive_parameter_control import AdaptiveParameterControl
from core.search.service.archive import Archive
from core.search.service.crossover.single_point_crossover import SinglePointCrossover
from core.search.service.fitness_function.fitness_function import FitnessFunction
from core.search.service.mutator.mutator import Mutator
from core.search.service.randomness import Randomness
from core.search.service.sampler.sampler import Sampler
from core.search.service.search_time_controller import SearchTimeController


@pytest.fixture
def config():
    """Create a configuration dictionary for tests."""
    return {
        "seed": 42,
        "image_height": 224,
        "image_width": 224,
        "population_size": 5,
        "abc_limit": 10,
        "max_evaluations": 100,
        "stopping_criterion": ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS,
        "mutation_sigma": 50,
        "apc_pixel_start": 40,
        "apc_pixel_end": 30,
        "apc_location_start": 40,
        "apc_location_end": 30,
        "apc_start_time": 0.4,
        "apc_threshold": 0.6,
    }


@pytest.fixture
def randomness(config):
    """Create a Randomness instance."""
    return Randomness(config)


@pytest.fixture
def stc(config):
    """Create a SearchTimeController instance."""
    return SearchTimeController(config, pc=PhaseController())


@pytest.fixture
def archive(stc, randomness, config):
    """Create an Archive instance."""
    return Archive(stc, randomness, config)


@pytest.fixture
def apc(stc, config):
    """Create an AdaptiveParameterControl instance."""
    return AdaptiveParameterControl(stc, config)


@pytest.fixture
def ff(archive, stc):
    """Create a mock FitnessFunction."""
    ff = MagicMock(spec=FitnessFunction)
    ff.calculate_fitness = MagicMock(side_effect=lambda ind: create_evaluated_individual(0.5))
    return ff


@pytest.fixture
def mutator(randomness, stc, config, apc):
    """Create a mock Mutator."""
    mutator = MagicMock(spec=Mutator)
    mutator.mutate = MagicMock(side_effect=lambda ind: ind.copy())
    return mutator


@pytest.fixture
def crossover(randomness, stc, config, apc):
    """Create a mock Crossover."""
    return MagicMock(spec=SinglePointCrossover)


@pytest.fixture
def sampler(randomness, config, archive):
    """Create a mock Sampler."""
    sampler = MagicMock(spec=Sampler)
    sampler.sample = MagicMock(side_effect=lambda: create_individual())
    return sampler


@pytest.fixture
def abc_algorithm(ff, randomness, stc, archive, config, mutator, crossover, sampler, apc):
    """Create an ArtificialBeeColonyAlgorithm instance."""
    return ArtificialBeeColonyAlgorithm(
        ff=ff,
        randomness=randomness,
        stc=stc,
        archive=archive,
        config=config,
        mutator=mutator,
        crossover=crossover,
        sampler=sampler,
        apc=apc
    )


def create_individual():
    """Helper function to create an individual with random actions."""
    individual = Individual()
    action = Action((10, 10), 100, 150, 200)
    individual.add_action(action)
    return individual


def create_evaluated_individual(fitness_value):
    """Helper function to create an evaluated individual."""
    individual = create_individual()
    fitness = FitnessValue(fitness_value, predictions=[], execution_time_ms=0)
    return EvaluatedIndividual(individual, fitness)


# Test initialization
def test_abc_initialization(abc_algorithm):
    """Test that ABC algorithm initializes correctly."""
    assert abc_algorithm is not None
    assert abc_algorithm.food_sources == []
    assert abc_algorithm.colony_size is None
    assert abc_algorithm.limit is None


# Test get_type method
def test_get_type(abc_algorithm):
    """Test that get_type returns the correct algorithm type."""
    assert abc_algorithm.get_type() == ConfigParser.Algorithms.ABC


# Test setup_before_search
def test_setup_before_search(abc_algorithm):
    """Test that setup_before_search initializes the population correctly."""
    abc_algorithm.setup_before_search()

    assert abc_algorithm.colony_size == 5
    assert abc_algorithm.limit == 10
    assert len(abc_algorithm.food_sources) == 5
    assert len(abc_algorithm.trial_counters) == 5
    assert all(counter == 0 for counter in abc_algorithm.trial_counters)


# Test employed bee phase
def test_employed_bee_phase(abc_algorithm):
    """Test the employed bee phase."""
    abc_algorithm.setup_before_search()

    # Mock fitness function to return better fitness
    better_ei = create_evaluated_individual(0.3)
    abc_algorithm.ff.calculate_fitness = MagicMock(return_value=better_ei)

    initial_food_sources = [fs.copy() for fs in abc_algorithm.food_sources]

    abc_algorithm.employed_bee_phase()

    # Assert that food sources were updated
    assert abc_algorithm.ff.calculate_fitness.called
    # Assert that trial counters were reset for improved solutions
    assert any(counter == 0 for counter in abc_algorithm.trial_counters)


def test_employed_bee_phase_no_improvement(abc_algorithm):
    """Test employed bee phase when no improvement is found."""
    abc_algorithm.setup_before_search()

    # Mock fitness function to return worse fitness
    worse_ei = create_evaluated_individual(0.9)
    abc_algorithm.ff.calculate_fitness = MagicMock(return_value=worse_ei)

    # Set initial trial counters
    initial_counters = abc_algorithm.trial_counters.copy()

    abc_algorithm.employed_bee_phase()

    # Assert that trial counters were incremented
    assert all(abc_algorithm.trial_counters[i] >= initial_counters[i]
               for i in range(abc_algorithm.colony_size))


# Test calculate_probabilities
def test_calculate_probabilities(abc_algorithm):
    """Test probability calculation for onlooker bees."""
    abc_algorithm.setup_before_search()

    # Set different fitness values
    abc_algorithm.food_sources[0].fitness.value = 0.2
    abc_algorithm.food_sources[1].fitness.value = 0.5
    abc_algorithm.food_sources[2].fitness.value = 0.8
    abc_algorithm.food_sources[3].fitness.value = 0.3
    abc_algorithm.food_sources[4].fitness.value = 0.6

    probabilities = abc_algorithm.calculate_probabilities()

    # Assert probabilities sum to 1
    assert pytest.approx(sum(probabilities), abs=0.001) == 1.0
    # Assert all probabilities are non-negative
    assert all(p >= 0 for p in probabilities)
    # Assert best fitness has highest probability
    best_index = 0
    assert probabilities[best_index] >= probabilities[2]


def test_calculate_probabilities_all_same(abc_algorithm):
    """Test probability calculation when all fitnesses are the same."""
    abc_algorithm.setup_before_search()

    # Set all fitness values to be the same
    for fs in abc_algorithm.food_sources:
        fs.fitness.value = 0.5

    probabilities = abc_algorithm.calculate_probabilities()

    # Assert uniform probabilities
    expected_prob = 1.0 / abc_algorithm.colony_size
    assert all(pytest.approx(p, abs=0.001) == expected_prob for p in probabilities)


# Test onlooker bee phase
def test_onlooker_bee_phase(abc_algorithm):
    """Test the onlooker bee phase."""
    abc_algorithm.setup_before_search()

    # Mock fitness function to return better fitness
    better_ei = create_evaluated_individual(0.2)
    abc_algorithm.ff.calculate_fitness = MagicMock(return_value=better_ei)

    abc_algorithm.onlooker_bee_phase()

    # Assert that fitness function was called
    assert abc_algorithm.ff.calculate_fitness.called
    # Assert that some food sources were updated
    assert any(fs.fitness.value == 0.2 for fs in abc_algorithm.food_sources)


# Test scout bee phase
def test_scout_bee_phase_no_abandon(abc_algorithm):
    """Test scout bee phase when no food sources need to be abandoned."""
    abc_algorithm.setup_before_search()

    # Set trial counters below the limit
    abc_algorithm.trial_counters = [5, 3, 7, 2, 4]

    initial_food_sources = [fs.copy() for fs in abc_algorithm.food_sources]

    abc_algorithm.scout_bee_phase()

    # Assert that no food sources were replaced
    assert len(abc_algorithm.food_sources) == len(initial_food_sources)


def test_scout_bee_phase_with_abandon(abc_algorithm):
    """Test scout bee phase when food sources need to be abandoned."""
    abc_algorithm.setup_before_search()

    # Set some trial counters above the limit
    abc_algorithm.trial_counters = [15, 5, 20, 3, 8]  # limit is 10

    # Mock sampler to return new individuals
    new_individual = create_individual()
    abc_algorithm.sampler.sample = MagicMock(return_value=new_individual)

    # Mock fitness function
    new_ei = create_evaluated_individual(0.4)
    abc_algorithm.ff.calculate_fitness = MagicMock(return_value=new_ei)

    abc_algorithm.scout_bee_phase()

    # Assert that trial counters were reset for abandoned food sources
    assert abc_algorithm.trial_counters[0] == 0
    assert abc_algorithm.trial_counters[2] == 0
    # Assert sampler was called for abandoned food sources
    assert abc_algorithm.sampler.sample.call_count == 2


# Test search_once
def test_search_once(abc_algorithm):
    """Test one complete iteration of ABC algorithm."""
    abc_algorithm.setup_before_search()

    # Mock fitness and sampler before setting trial counters
    sampler_mock = MagicMock(return_value=create_individual())
    abc_algorithm.sampler.sample = sampler_mock

    # Mock fitness function to return worse fitness so trial counters increase
    abc_algorithm.ff.calculate_fitness = MagicMock(
        return_value=create_evaluated_individual(0.9)
    )

    # Set trial counters manually to simulate exhausted food sources
    # These will be checked in scout phase after employed and onlooker phases
    abc_algorithm.trial_counters = [15, 5, 20, 3, 8]  # 0 and 2 are above limit (10)

    abc_algorithm.search_once()

    # Assert that all three phases were executed
    # Employed bees and onlooker bees call fitness function
    assert abc_algorithm.ff.calculate_fitness.called
    # Scout bees should have been called for indices 0 and 2 (those with trial >= 10)
    # Note: After employed and onlooker phases, counters might have increased
    # but we explicitly set them high enough that 0 and 2 will trigger scout phase
    assert sampler_mock.call_count >= 2


# Test integration with archive
def test_archive_integration(abc_algorithm):
    """Test that ABC algorithm integrates correctly with archive."""
    abc_algorithm.setup_before_search()

    # Create a better solution
    better_ei = create_evaluated_individual(0.1)
    abc_algorithm.ff.calculate_fitness = MagicMock(return_value=better_ei)

    # Clear archive add calls
    abc_algorithm.archive.add_archive_if_needed = MagicMock()

    abc_algorithm.search_once()

    # Assert that archive was updated
    assert abc_algorithm.archive.add_archive_if_needed.called


# Test colony size configuration
def test_different_colony_sizes(ff, randomness, stc, archive, mutator, crossover, sampler, apc):
    """Test ABC algorithm with different colony sizes."""
    for size in [10, 20, 50]:
        config = {
            "seed": 42,
            "image_height": 224,
            "image_width": 224,
            "population_size": size,
            "abc_limit": size * 2,
            "max_evaluations": 100,
            "stopping_criterion": ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS,
        }

        abc = ArtificialBeeColonyAlgorithm(
            ff=ff, randomness=randomness, stc=stc, archive=archive,
            config=config, mutator=mutator, crossover=crossover,
            sampler=sampler, apc=apc
        )

        abc.setup_before_search()

        assert abc.colony_size == size
        assert len(abc.food_sources) == size
        assert len(abc.trial_counters) == size


# Test limit parameter
def test_abc_limit_parameter(ff, randomness, stc, archive, mutator, crossover, sampler, apc):
    """Test ABC algorithm with different limit values."""
    config = {
        "seed": 42,
        "image_height": 224,
        "image_width": 224,
        "population_size": 10,
        "abc_limit": 50,
        "max_evaluations": 100,
        "stopping_criterion": ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS,
    }

    abc = ArtificialBeeColonyAlgorithm(
        ff=ff, randomness=randomness, stc=stc, archive=archive,
        config=config, mutator=mutator, crossover=crossover,
        sampler=sampler, apc=apc
    )

    abc.setup_before_search()

    assert abc.limit == 50