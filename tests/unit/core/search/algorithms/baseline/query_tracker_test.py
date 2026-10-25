import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.config_parser import ConfigParser
from core.search.algorithms.baseline.query_tracker import (
    STALL_LIMIT, AttackStalled, BudgetExhausted, QueryTracker, SolutionFound, StopAttack)
from core.search.phase_controller import PhaseController
from core.search.service.search_time_controller import SearchTimeController


def make_stc(max_evaluations=10, criterion=ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS):
    config = {"max_evaluations": max_evaluations, "stopping_criterion": criterion, "fitness_threshold": 0.0}
    stc = SearchTimeController(config, pc=PhaseController())
    stc.start_search()
    return stc


def make_tracker(stc, norm=2, stop_at_success=True, on_new_best=None):
    original = np.zeros((2, 2, 3), dtype=np.uint8)
    return QueryTracker(stc, lambda result: result.fitness, original, norm, stop_at_success,
                        on_new_best or MagicMock())


def img(*values):
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image.flat[:len(values)] = values
    return image


def res(fitness):
    result = MagicMock()
    result.fitness = fitness
    result.predictions = []
    return result


def test_exceptions_hierarchy():
    assert issubclass(BudgetExhausted, StopAttack)
    assert issubclass(SolutionFound, StopAttack)


def test_before_query_raises_when_budget_is_used():
    stc = make_stc(2)
    tracker = make_tracker(stc)
    tracker.before_query()
    stc.new_individual_evaluation()
    tracker.before_query()
    stc.new_individual_evaluation()
    with pytest.raises(BudgetExhausted):
        tracker.before_query()


def test_before_query_respects_time_budget():
    stc = make_stc(0.01, ConfigParser.StoppingCriterion.TIME)
    tracker = make_tracker(stc)
    time.sleep(0.05)
    with pytest.raises(BudgetExhausted):
        tracker.before_query()


def test_stops_at_first_success_when_requested():
    tracker = make_tracker(make_stc())
    tracker.after_query(img(5), res(0.5))
    success = img(1)
    with pytest.raises(SolutionFound):
        tracker.after_query(success, res(0.0))
    assert np.array_equal(tracker.best_image, success)
    assert tracker.best_fitness.value == 0.0


def test_keeps_smallest_successful_perturbation_when_not_stopping():
    tracker = make_tracker(make_stc(), norm=2, stop_at_success=False)
    small = img(3)
    tracker.after_query(img(10), res(0.0))
    tracker.after_query(small, res(0.0))
    tracker.after_query(img(20), res(-0.5))  # lower fitness but larger perturbation
    assert np.array_equal(tracker.best_image, small)
    assert tracker.best_fitness.value == 0.0
    tracker.after_query(img(1), res(0.7))  # unsuccessful never beats a success
    assert np.array_equal(tracker.best_image, small)


def test_before_any_success_best_is_lowest_fitness():
    tracker = make_tracker(make_stc(), stop_at_success=False)
    tracker.after_query(img(1), res(0.8))
    low = img(9)
    tracker.after_query(low, res(0.3))
    tracker.after_query(img(2), res(0.6))
    assert np.array_equal(tracker.best_image, low)
    assert tracker.best_fitness.value == 0.3


def test_on_new_best_called_only_when_best_changes():
    callback = MagicMock()
    tracker = make_tracker(make_stc(), stop_at_success=False, on_new_best=callback)
    tracker.after_query(img(1), res(0.8))
    tracker.after_query(img(2), res(0.9))
    tracker.after_query(img(3), res(0.5))
    assert callback.call_count == 2
    image, fitness = callback.call_args[0]
    assert np.array_equal(image, img(3))
    assert fitness.value == 0.5


def test_l0_distance_counts_changed_pixels():
    tracker = make_tracker(make_stc(), norm=0)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = [1, 2, 3]
    image[1, 1, 2] = 200
    assert tracker.distance(image) == 2


def test_distance_does_not_wrap_uint8():
    stc = make_stc()
    original = np.full((1, 1, 3), 10, dtype=np.uint8)
    tracker = QueryTracker(stc, lambda r: 0.0, original, float("inf"), True, MagicMock())
    assert tracker.distance(np.zeros((1, 1, 3), dtype=np.uint8)) == 10


def test_query_worse_than_original_does_not_become_best():
    on_new_best = MagicMock()
    tracker = make_tracker(make_stc(), on_new_best=on_new_best)
    tracker.start_from_original(res(0.5))
    tracker.after_query(img(1), res(0.7))
    on_new_best.assert_not_called()
    assert np.array_equal(tracker.best_image, tracker.original)
    tracker.after_query(img(2), res(0.4))
    on_new_best.assert_called_once()
    assert tracker.best_fitness.value == 0.4


def test_cache_hits_without_new_query_stop_the_attack():
    tracker = make_tracker(make_stc())
    for _ in range(STALL_LIMIT - 1):
        tracker.cache_hit()
    tracker.after_query(img(1), res(0.5))
    for _ in range(STALL_LIMIT - 1):
        tracker.cache_hit()
    with pytest.raises(AttackStalled):
        tracker.cache_hit()
    assert issubclass(AttackStalled, StopAttack)
