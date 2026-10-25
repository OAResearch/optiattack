"""Every algorithm must respect the evaluation budget and stop as soon as an adversarial example is found."""
import pytest

from core.config_parser import ConfigParser
from core.search.algorithms.abc_algorithm import ArtificialBeeColonyAlgorithm
from core.search.algorithms.genetic_algorithm import GeneticAlgorithm
from core.search.algorithms.mabc_algorithm import ModifiedArtificialBeeColonyAlgorithm
from core.search.algorithms.mio_algorithm import MioAlgorithm
from core.search.fitness_value import FitnessValue
from core.search.phase_controller import PhaseController
from core.search.service.adaptive_parameter_control import AdaptiveParameterControl
from core.search.service.archive import Archive
from core.search.service.crossover.single_point_crossover import SinglePointCrossover
from core.search.service.fitness_function.fitness_function import FitnessFunction
from core.search.service.mutator.standard_mutator import StandardMutator
from core.search.service.randomness import Randomness
from core.search.service.sampler.random_sampler import RandomSampler
from core.search.service.search_time_controller import SearchTimeController

ALGORITHMS = [MioAlgorithm, GeneticAlgorithm, ArtificialBeeColonyAlgorithm, ModifiedArtificialBeeColonyAlgorithm]


class ScriptedFitnessFunction(FitnessFunction):

    """Counts every evaluation as a NUT query and reports success only at a given evaluation."""

    def __init__(self, archive, stc, success_at=None):
        super().__init__(archive, None, stc)
        self.calls = 0
        self.success_at = success_at

    def evaluate(self, individual=None, actions=None):
        self.calls += 1
        self.stc.new_individual_evaluation()
        return FitnessValue(-0.1 if self.calls == self.success_at else 0.5, [])


def build_algorithm(algorithm_class, success_at=None, max_evaluations=25):
    config = {
        "seed": 1,
        "image_height": 16,
        "image_width": 16,
        "min_action_size": 1,
        "max_action_size": 3,
        "population_size": 10,
        "abc_limit": 100,
        "max_evaluations": max_evaluations,
        "stopping_criterion": ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS,
        "fitness_threshold": 0.0,
        "random_sampling_probability": 0.5,
        "focused_search_activation_time": 0.8,
        "apc_pixel_start": 40,
        "apc_pixel_end": 30,
        "apc_location_start": 4,
        "apc_location_end": 2,
        "apc_start_time": 0.4,
        "apc_threshold": 0.6,
    }
    randomness = Randomness(config)
    stc = SearchTimeController(config, pc=PhaseController())
    apc = AdaptiveParameterControl(stc, config)
    archive = Archive(stc, randomness, config)
    ff = ScriptedFitnessFunction(archive, stc, success_at)
    algorithm = algorithm_class(ff=ff, randomness=randomness, stc=stc, archive=archive, config=config,
                                mutator=StandardMutator(randomness, stc, config, apc),
                                crossover=SinglePointCrossover(randomness, stc, config, apc),
                                sampler=RandomSampler(randomness, archive, config), apc=apc)
    return algorithm, stc


@pytest.mark.parametrize("algorithm_class", ALGORITHMS)
def test_search_never_exceeds_the_evaluation_budget(algorithm_class):
    algorithm, stc = build_algorithm(algorithm_class, max_evaluations=25)

    algorithm.search()

    assert stc.get_evaluated_individuals() == 25


@pytest.mark.parametrize("algorithm_class", ALGORITHMS)
@pytest.mark.parametrize("success_at", [7, 12, 13])
def test_search_stops_at_the_evaluation_that_finds_the_adversarial_example(algorithm_class, success_at):
    # 7 falls inside population initialisation, 12 and 13 inside the first generation
    algorithm, stc = build_algorithm(algorithm_class, success_at=success_at, max_evaluations=1000)

    algorithm.search()

    assert stc.get_evaluated_individuals() == success_at
    assert stc.get_current_fitness_value() == -0.1
