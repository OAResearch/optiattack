"""With --max_pixels, no OptiAttack algorithm may send the NUT an image that changes more than k pixels."""
import numpy as np
import pytest

from core.config_parser import ConfigParser
from core.search.action import Action
from core.search.algorithms.abc_algorithm import ArtificialBeeColonyAlgorithm
from core.search.algorithms.genetic_algorithm import GeneticAlgorithm
from core.search.algorithms.mabc_algorithm import ModifiedArtificialBeeColonyAlgorithm
from core.search.algorithms.mio_algorithm import MioAlgorithm
from core.search.algorithms.random_algorithm import RandomAlgorithm
from core.search.evaluated_individual import EvaluatedIndividual
from core.search.fitness_value import FitnessValue
from core.search.individual import Individual
from core.search.phase_controller import PhaseController
from core.search.service.adaptive_parameter_control import AdaptiveParameterControl
from core.search.service.archive import Archive
from core.search.service.crossover.single_point_crossover import SinglePointCrossover
from core.search.service.fitness_function.targeted_fitness_function import TargetedFitnessFunction
from core.search.service.fitness_function.untargeted_fitness_function import UntargetedFitnessFunction
from core.search.service.mutator.standard_mutator import StandardMutator
from core.search.service.pixel_budget import PixelBudget
from core.search.service.randomness import Randomness
from core.search.service.sampler.random_sampler import RandomSampler
from core.search.service.search_time_controller import SearchTimeController
from core.utils.images import ProcessedImage
from core.utils.nut_request import NutRequest
from core.utils.perturbation import l0_pixels

K = 5
SIZE = 8
LABELS = ["dark", "bright", "mid"]
ALGORITHMS = {"random": RandomAlgorithm, "mio": MioAlgorithm, "genetic": GeneticAlgorithm,
              "abc": ArtificialBeeColonyAlgorithm, "mabc": ModifiedArtificialBeeColonyAlgorithm}


def make_image():
    rng = np.random.RandomState(0)
    return rng.randint(60, 200, size=(SIZE, SIZE, 3)).astype(np.uint8)


class FakeNut:

    """Fake remote controller that records every image; the more the image changes, the better (never flips)."""

    def __init__(self, stc, original):
        self.stc = stc
        self.original = original.astype(float)
        self.images = []
        self.target = None

    def response(self, image):
        change = float(np.mean(np.abs(np.asarray(image, dtype=float) - self.original))) / 255.0
        scores = {"dark": 0.8 - 0.3 * change, "bright": 0.15 + 0.3 * change, "mid": 0.05}
        return {"predictions": [{"label": label, "score": scores[label]} for label in LABELS]}

    def new_action(self, image):
        self.stc.new_individual_evaluation()
        self.images.append(np.array(image, copy=True))
        return NutRequest(self.response(image), self.target)


def make_setup(max_pixels=K, max_evaluations=60, seed=1):
    config = dict(ConfigParser().default_params())
    config.update({"seed": seed, "image_height": SIZE, "image_width": SIZE, "min_action_size": 1,
                   "max_action_size": 3, "population_size": 10, "abc_limit": 100,
                   "max_evaluations": max_evaluations, "max_pixels": max_pixels,
                   "stopping_criterion": ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS})
    randomness = Randomness(config)
    stc = SearchTimeController(config, pc=PhaseController())
    archive = Archive(stc, randomness, config)
    image = make_image()
    archive.set_image(ProcessedImage(None, None, image))
    nut = FakeNut(stc, image)
    archive.set_original_prediction_results(nut.response(image))
    ff = UntargetedFitnessFunction(archive, nut, stc, pixel_budget=PixelBudget(config, randomness, archive))
    return config, randomness, stc, archive, nut, ff


def make_algorithm(name, **kwargs):
    config, randomness, stc, archive, nut, ff = make_setup(**kwargs)
    apc = AdaptiveParameterControl(stc, config)
    algorithm = ALGORITHMS[name](ff=ff, randomness=randomness, stc=stc, archive=archive, config=config,
                                 mutator=StandardMutator(randomness, stc, config, apc),
                                 crossover=SinglePointCrossover(randomness, stc, config, apc),
                                 sampler=RandomSampler(randomness, archive, config), apc=apc)
    return algorithm, nut


def bounded_search(algorithm, max_iterations=20000):
    """algorithm.search() with an iteration cap, so that a search that stops querying fails instead of hanging."""
    algorithm.stc.start_search()
    algorithm.setup_before_search()
    iterations = 0
    while algorithm.stc.should_continue_search() and iterations < max_iterations:
        algorithm.search_once()
        iterations += 1
    return algorithm.after_search()


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_never_sends_more_than_k_pixels(name):
    algorithm, nut = make_algorithm(name)
    solution = bounded_search(algorithm)

    original = algorithm.archive.image.array
    assert nut.images
    assert max(l0_pixels(original, img) for img in nut.images) <= K
    assert l0_pixels(original, algorithm.archive.get_mutated_image(solution.actions)) <= K


def test_mio_keeps_searching_when_archive_is_full():
    algorithm, nut = make_algorithm("mio")
    solution = bounded_search(algorithm)

    original = algorithm.archive.image.array
    assert algorithm.stc.get_evaluated_individuals() == 60
    assert len(nut.images) == 60
    # The archive really was full: the run kept spending its budget on recolouring the k pixels.
    assert l0_pixels(original, algorithm.archive.get_mutated_image(solution.actions)) == K
    assert max(l0_pixels(original, img) for img in nut.images) <= K


@pytest.mark.parametrize("name", ["random", "genetic", "abc", "mabc"])
def test_other_algorithms_keep_searching_when_archive_is_full(name):
    algorithm, nut = make_algorithm(name)
    solution = bounded_search(algorithm, max_iterations=200)
    original = algorithm.archive.image.array
    assert algorithm.stc.get_evaluated_individuals() == 60
    assert l0_pixels(original, algorithm.archive.get_mutated_image(solution.actions)) == K


def test_genetic_keeps_searching_when_archive_is_full():
    algorithm, nut = make_algorithm("genetic")
    solution = algorithm.search()
    original = algorithm.archive.image.array
    assert algorithm.stc.get_evaluated_individuals() == 60
    assert len(nut.images) == 60
    assert max(l0_pixels(original, img) for img in nut.images) <= K
    assert l0_pixels(original, algorithm.archive.get_mutated_image(solution.actions)) == K


def test_empty_individual_is_not_queried():
    _config, _randomness, stc, archive, nut, ff = make_setup()
    full = Individual()
    full.actions = [Action((0, c), 0, 0, 0) for c in range(K)]
    archive.populations = [EvaluatedIndividual(full, FitnessValue(0.5, []))]
    candidate = Individual()

    ei = ff.calculate_fitness(candidate)

    original = archive.get_original_prediction_results()
    assert stc.get_evaluated_individuals() == 0
    assert nut.images == []
    assert ei.individual is candidate
    assert ei.fitness.value == ff.fitness_from_result(original)
    assert ei.fitness.predictions is original.predictions


def test_without_budget_fitness_function_still_queries_everything():
    _config, _randomness, stc, archive, nut, _ff = make_setup()
    ff = UntargetedFitnessFunction(archive, nut, stc)
    candidate = Individual()
    candidate.actions = [Action((r, c), 0, 0, 0) for r in range(3) for c in range(SIZE)]

    ff.calculate_fitness(candidate)

    assert stc.get_evaluated_individuals() == 1
    assert l0_pixels(archive.image.array, nut.images[0]) == 3 * SIZE


def test_targeted_fitness_function_repairs_too():
    _config, randomness, stc, archive, nut, _ff = make_setup()
    budget = PixelBudget({"max_pixels": 2}, randomness, archive)
    nut.target = "bright"
    ff = TargetedFitnessFunction(archive, nut, stc, "bright", pixel_budget=budget)
    candidate = Individual()
    candidate.actions = [Action((r, 0), 0, 0, 0) for r in range(6)]

    ff.calculate_fitness(candidate)

    assert ff.target == "bright"
    assert len(candidate.actions) == 6
    assert l0_pixels(archive.image.array, nut.images[0]) == 2


@pytest.mark.parametrize("attack_type", [ConfigParser.AttackType.UNTARGETED, ConfigParser.AttackType.TARGETED])
def test_application_wires_pixel_budget(attack_type):
    from core.problem.base_module import BaseModule
    from core.utils.application import configure_container

    container = BaseModule()
    config = dict(ConfigParser().default_params())
    config.update({"algorithm": ConfigParser.Algorithms.MIO, "attack_type": attack_type, "target": "bright",
                   "max_pixels": 7})
    container.config.override(config)
    container = configure_container(container)
    ff = container.ff()
    assert ff.pixel_budget is container.pixel_budget()
    assert ff.pixel_budget.limit == 7
    assert ff.pixel_budget.archive is container.archive()
    assert container.algorithm().ff is ff
