from unittest.mock import MagicMock

import numpy as np
import pytest

from core.config_parser import ConfigParser
from core.search.algorithms.baseline.query_tracker import BudgetExhausted, QueryTracker
from core.search.phase_controller import PhaseController
from core.search.service.search_time_controller import SearchTimeController
from core.utils.nut_request import NutRequest

pytest.importorskip("art")

from core.search.algorithms.baseline.estimator import RemoteNutEstimator  # noqa: E402

LABELS = ["cat", "dog", "bird"]
SHAPE = (2, 2, 3)


class FakeRemoteController:

    def __init__(self, stc, scores=(0.2, 0.7, 0.1)):
        self.stc = stc
        self.scores = scores
        self.received = []

    def new_action(self, image):
        self.received.append(np.array(image, copy=True))
        self.stc.new_individual_evaluation()
        return NutRequest({"predictions": [{"label": label, "score": score}
                                           for label, score in zip(LABELS, self.scores)]})


def make_stc(max_evaluations=100):
    config = {"max_evaluations": max_evaluations,
              "stopping_criterion": ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS,
              "fitness_threshold": 0.0}
    stc = SearchTimeController(config, pc=PhaseController())
    stc.start_search()
    return stc


def make_estimator(max_evaluations=100, label_only=False):
    stc = make_stc(max_evaluations)
    remote = FakeRemoteController(stc)
    original = np.zeros(SHAPE, dtype=np.uint8)
    tracker = QueryTracker(stc, lambda result: -1.0, original, 2, False, MagicMock())
    return RemoteNutEstimator(remote, LABELS, SHAPE, tracker, label_only=label_only), remote


def test_predict_returns_probability_vectors_for_each_image():
    estimator, remote = make_estimator()
    x = np.arange(3, dtype=np.float32).reshape(3, 1, 1, 1) * np.ones((3,) + SHAPE, dtype=np.float32)
    preds = estimator.predict(x)
    assert preds.shape == (3, 3)
    np.testing.assert_allclose(preds, [[0.2, 0.7, 0.1]] * 3, atol=1e-6)
    assert len(remote.received) == 3


def test_predict_sends_rounded_and_clipped_uint8_images():
    estimator, remote = make_estimator()
    x = np.zeros((1,) + SHAPE, dtype=np.float32)
    x.flat[:3] = [-3.0, 255.7, 10.4]
    estimator.predict(x)
    sent = remote.received[0]
    assert sent.dtype == np.uint8
    assert sent.shape == SHAPE
    assert list(sent.flat[:3]) == [0, 255, 10]


def test_label_only_returns_one_hot():
    estimator, _ = make_estimator(label_only=True)
    preds = estimator.predict(np.zeros((2,) + SHAPE, dtype=np.float32))
    np.testing.assert_array_equal(preds, [[0, 1, 0]] * 2)


def test_predict_stops_inside_a_batch_when_budget_runs_out():
    estimator, remote = make_estimator(max_evaluations=2)
    with pytest.raises(BudgetExhausted):
        estimator.predict(np.arange(4, dtype=np.float32).reshape(4, 1, 1, 1) * np.ones((4,) + SHAPE, dtype=np.float32))
    assert len(remote.received) == 2


def test_repeated_image_is_sent_once_and_counted_once():
    estimator, remote = make_estimator()
    x = np.full((1,) + SHAPE, 10.0, dtype=np.float32)
    first = estimator.predict(x)
    again = estimator.predict(np.concatenate([x, x + 0.2]))  # same image after rounding to uint8
    assert len(remote.received) == 1
    assert remote.stc.get_evaluated_individuals() == 1
    np.testing.assert_allclose(again, np.concatenate([first, first]))


def test_cache_hit_does_not_need_budget():
    estimator, remote = make_estimator(max_evaluations=1)
    x = np.zeros((1,) + SHAPE, dtype=np.float32)
    estimator.predict(x)
    estimator.predict(x)
    assert len(remote.received) == 1
    with pytest.raises(BudgetExhausted):
        estimator.predict(x + 1)


def test_cache_hit_does_not_reach_the_tracker():
    estimator, _ = make_estimator()
    estimator.tracker.after_query = MagicMock(wraps=estimator.tracker.after_query)
    x = np.zeros((1,) + SHAPE, dtype=np.float32)
    estimator.predict(x)
    estimator.predict(x)
    assert estimator.tracker.after_query.call_count == 1


def test_label_only_cache_hit_returns_one_hot():
    estimator, remote = make_estimator(label_only=True)
    x = np.zeros((1,) + SHAPE, dtype=np.float32)
    estimator.predict(x)
    np.testing.assert_array_equal(estimator.predict(x), [[0, 1, 0]])
    assert len(remote.received) == 1


def test_cache_is_bounded_and_evicts_least_recently_used():
    estimator, remote = make_estimator()
    estimator.cache_size = 2
    images = [np.full((1,) + SHAPE, v, dtype=np.float32) for v in (1, 2, 3)]
    estimator.predict(images[0])
    estimator.predict(images[1])
    estimator.predict(images[0])  # hit, 0 becomes most recent
    estimator.predict(images[2])  # evicts 1
    estimator.predict(images[0])  # still cached
    assert len(remote.received) == 3
    estimator.predict(images[1])  # evicted, queried again
    assert len(remote.received) == 4


def test_seeded_image_is_answered_from_the_cache_without_query():
    estimator, remote = make_estimator()
    original = np.full(SHAPE, 7, dtype=np.uint8)
    result = NutRequest({"predictions": [{"label": "cat", "score": 0.6}, {"label": "dog", "score": 0.3}]})
    estimator.seed_cache(original, result)
    preds = estimator.predict(original[np.newaxis].astype(np.float32))
    assert remote.received == []
    assert remote.stc.get_evaluated_individuals() == 0
    np.testing.assert_allclose(preds, [[0.6, 0.3, 0.1]], atol=1e-6)
