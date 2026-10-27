import numpy as np
import pytest

from core.search.action import Action
from core.search.evaluated_individual import EvaluatedIndividual
from core.search.fitness_value import FitnessValue
from core.search.individual import Individual
from core.search.phase_controller import PhaseController
from core.search.service.archive import Archive
from core.search.service.pixel_budget import PixelBudget
from core.search.service.randomness import Randomness
from core.search.service.search_time_controller import SearchTimeController
from core.utils.images import ProcessedImage
from core.utils.perturbation import l0_pixels

ORIGINAL_VALUE = 100


def make_budget(limit, archive_locations=(), seed=3):
    config = {"seed": seed, "max_pixels": limit}
    randomness = Randomness(config)
    stc = SearchTimeController(config, pc=PhaseController())
    archive = Archive(stc, randomness, config)
    archive.set_image(ProcessedImage(None, None, np.full((6, 6, 3), ORIGINAL_VALUE, dtype=np.uint8)))
    if archive_locations:
        individual = make_individual(archive_locations, color=(0, 0, 0))
        archive.populations = [EvaluatedIndividual(individual, FitnessValue(0.5, []))]
    return PixelBudget(config, randomness, archive), archive


def make_individual(locations, color=(255, 255, 255)):
    individual = Individual()
    for location in locations:
        individual.actions.append(Action(location, *color))
    return individual


def changed_pixels(archive, individual):
    image = individual.get_action_image(archive.get_mutated_image())
    return l0_pixels(archive.image.array, image)


def test_repair_keeps_individual_within_limit():
    budget, archive = make_budget(3, archive_locations=[(0, 0), (0, 1)])
    individual = make_individual([(2, 2), (3, 3), (4, 4), (5, 5)])

    assert budget.repair(individual) is True

    assert changed_pixels(archive, individual) <= 3
    assert len(individual.actions) == 4


def test_repair_relocates_instead_of_dropping():
    budget, archive = make_budget(3, archive_locations=[(0, 0), (0, 1)])
    individual = make_individual([(2, 2), (3, 3), (4, 4), (5, 5)], color=(7, 8, 9))

    assert budget.repair(individual) is True

    locations = [a.get_location() for a in individual.actions]
    kept_new = [loc for loc in locations if loc not in [(0, 0), (0, 1)]]
    assert len(individual.actions) == 4
    assert all(list(a.get_color()) == [7, 8, 9] for a in individual.actions)
    # One new location fits the budget; the three others were moved onto already-changed pixels.
    assert len(set(kept_new)) == 1
    assert set(locations) <= {(0, 0), (0, 1)} | set(kept_new)
    assert changed_pixels(archive, individual) == 3


def test_repair_keeps_actions_on_already_changed_pixels():
    budget, archive = make_budget(3, archive_locations=[(0, 0), (0, 1)])
    on_archive = [Action((0, 0), 10, 20, 30), Action((0, 1), 40, 50, 60)]
    unchanged = Action((1, 1), ORIGINAL_VALUE, ORIGINAL_VALUE, ORIGINAL_VALUE)
    individual = make_individual([(2, 2), (3, 3), (4, 4)])
    individual.actions = on_archive[:1] + individual.actions[:2] + [unchanged] + on_archive[1:] + \
        individual.actions[2:]

    assert budget.repair(individual) is True

    for action in on_archive + [unchanged]:
        assert any(a is action for a in individual.actions)
    assert len(individual.actions) == 6
    assert changed_pixels(archive, individual) == 3


def test_repeated_location_counts_once():
    budget, archive = make_budget(2)
    individual = make_individual([(2, 2), (2, 2), (3, 3)])

    assert budget.repair(individual) is True

    assert len(individual.actions) == 3
    assert changed_pixels(archive, individual) == 2


def test_repair_moves_everything_onto_archive_pixels_when_archive_is_full():
    budget, archive = make_budget(3, archive_locations=[(0, 0), (0, 1), (0, 2)])
    individual = make_individual([(2, 2), (3, 3)])

    assert budget.repair(individual) is True

    assert len(individual.actions) == 2
    assert {a.get_location() for a in individual.actions} <= {(0, 0), (0, 1), (0, 2)}
    assert changed_pixels(archive, individual) == 3


def test_repair_returns_false_only_for_an_individual_without_actions():
    budget, archive = make_budget(3, archive_locations=[(0, 0), (0, 1), (0, 2)])
    assert budget.repair(Individual()) is False


def test_limit_zero_means_unlimited():
    budget, archive = make_budget(0, archive_locations=[(0, 0), (0, 1)])
    locations = [(r, c) for r in range(1, 6) for c in range(6)]
    individual = make_individual(locations)

    assert budget.limit == 0
    assert budget.repair(individual) is True

    assert len(individual.actions) == len(locations)


def test_repair_is_seeded():
    kept = []
    for _ in range(2):
        budget, _archive = make_budget(4, archive_locations=[(0, 0)], seed=11)
        individual = make_individual([(r, c) for r in range(1, 6) for c in range(6)])
        budget.repair(individual)
        kept.append([a.get_location() for a in individual.actions])
    assert len(kept[0]) == 30
    assert len(set(kept[0])) == 4
    assert kept[0] == kept[1]


def test_negative_limit_is_rejected():
    with pytest.raises(ValueError, match="max_pixels"):
        make_budget(-1)


def test_archive_above_limit_raises_instead_of_sending_a_larger_image():
    budget, archive = make_budget(2, archive_locations=[(0, 0), (0, 1), (0, 2)])
    with pytest.raises(RuntimeError, match="max_pixels"):
        budget.repair(make_individual([(0, 0)]))
