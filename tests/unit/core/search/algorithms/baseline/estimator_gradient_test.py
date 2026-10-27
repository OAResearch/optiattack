from unittest.mock import MagicMock

import numpy as np
import pytest

from core.config_parser import ConfigParser
from core.search.algorithms.baseline.nut_interface import (InvalidGradientError, NutCapabilityError,
                                                           to_logit_vector)
from core.search.algorithms.baseline import query_tracker
from core.search.algorithms.baseline.query_tracker import AttackStalled, BudgetExhausted, QueryTracker
from core.search.phase_controller import PhaseController
from core.search.service.search_time_controller import SearchTimeController
from core.utils.nut_request import NutRequest

pytest.importorskip("art")

from art.attacks.evasion import CarliniL2Method, FastGradientMethod, ProjectedGradientDescent  # noqa: E402

from core.search.algorithms.baseline.estimator import RemoteNutEstimator  # noqa: E402

LABELS = ["cat", "dog", "bird", "fish", "frog", "deer"]
SHAPE = (2, 2, 3)
RNG = np.random.default_rng(0)
W = RNG.normal(scale=0.01, size=(len(LABELS), int(np.prod(SHAPE))))
B = RNG.normal(scale=0.1, size=len(LABELS))


def logits_of(image):
    return W @ np.asarray(image, dtype=np.float64).ravel() + B


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


def analytic_loss_gradient(image, index):
    probs = softmax(logits_of(image))
    probs[index] -= 1.0
    return (probs @ W).reshape(SHAPE)


class LinearNut:

    """Fake remote controller of a linear model: logits = W @ x.ravel() + b, with analytic gradients."""

    def __init__(self, stc, with_logits=True, bad_gradient=None):
        self.stc = stc
        self.with_logits = with_logits
        self.bad_gradient = bad_gradient
        self.gradient_calls = []
        self.received = []

    def response(self, image):
        logits = logits_of(image)
        probs = softmax(logits)
        top5 = np.argsort(-probs)[:5]
        result = {"predictions": [{"label": LABELS[i], "score": float(probs[i])} for i in top5]}
        if self.with_logits:
            result["logits"] = [float(v) for v in logits]
        return result

    def new_action(self, image):
        self.received.append(np.array(image, copy=True))
        self.stc.new_individual_evaluation()
        return NutRequest(self.response(image))

    def gradient(self, image, label, kind):
        self.gradient_calls.append((np.array(image, copy=True), label, kind))
        self.stc.new_individual_evaluation()
        self.stc.new_gradient_evaluation()
        if self.bad_gradient is not None:
            return self.bad_gradient
        index = LABELS.index(label)
        if kind == "class":
            return W[index].reshape(SHAPE).astype(np.float32)
        return analytic_loss_gradient(image, index).astype(np.float32)


def make_stc(max_evaluations=100):
    config = {"max_evaluations": max_evaluations,
              "stopping_criterion": ConfigParser.StoppingCriterion.INDIVIDUAL_EVALUATIONS,
              "fitness_threshold": 0.0}
    stc = SearchTimeController(config, pc=PhaseController())
    stc.start_search()
    return stc


def make_estimator(max_evaluations=100, use_logits=False, fitness=lambda result: 1.0, allow_gradients=True,
                   **nut_kwargs):
    stc = make_stc(max_evaluations)
    nut = LinearNut(stc, **nut_kwargs)
    tracker = QueryTracker(stc, fitness, np.zeros(SHAPE, dtype=np.uint8), 2, False, MagicMock())
    return RemoteNutEstimator(nut, LABELS, SHAPE, tracker, use_logits=use_logits,
                              allow_gradients=allow_gradients), nut


def test_black_box_estimator_refuses_gradient_queries():
    # anti-goal 5: a black-box baseline must never be able to ask the NUT for gradients
    estimator, nut = make_estimator(allow_gradients=False)
    x = images(1)
    with pytest.raises(RuntimeError, match="white-box"):
        estimator.loss_gradient(x, np.array([1]))
    with pytest.raises(RuntimeError, match="white-box"):
        estimator.class_gradient(x, label=1)
    with pytest.raises(RuntimeError, match="white-box"):
        estimator.class_gradient(x)
    assert nut.gradient_calls == []
    assert estimator.tracker.stc.get_gradient_evaluations() == 0


def images(n, seed=1):
    return np.random.default_rng(seed).uniform(0, 255, size=(n,) + SHAPE).astype(np.float32)


def test_loss_gradient_matches_analytic():
    estimator, nut = make_estimator()
    x = images(2)
    one_hot = np.eye(len(LABELS))[[1, 4]]

    grads = estimator.loss_gradient(x, one_hot)

    assert grads.shape == x.shape
    assert grads.dtype == np.float32
    for i, index in enumerate([1, 4]):
        np.testing.assert_allclose(grads[i], analytic_loss_gradient(x[i], index), rtol=1e-4, atol=1e-7)
    assert [(label, kind) for _, label, kind in nut.gradient_calls] == [("dog", "loss"), ("frog", "loss")]
    np.testing.assert_allclose(estimator.loss_gradient(x, np.array([1, 4])), grads)  # index labels


def test_class_gradient_shapes_match_art_contract():
    estimator, _ = make_estimator()
    x = images(2)

    single = estimator.class_gradient(x, label=2)
    assert single.shape == (2, 1) + SHAPE
    np.testing.assert_allclose(single[:, 0], np.stack([W[2].reshape(SHAPE)] * 2), rtol=1e-6)

    per_sample = estimator.class_gradient(x, label=np.array([3, 5]))
    assert per_sample.shape == (2, 1) + SHAPE
    np.testing.assert_allclose(per_sample[1, 0], W[5].reshape(SHAPE), rtol=1e-6)
    np.testing.assert_allclose(estimator.class_gradient(x, label=[3, 5]), per_sample)

    every = estimator.class_gradient(x, label=None)
    assert every.shape == (2, len(LABELS)) + SHAPE
    assert every.dtype == np.float32
    np.testing.assert_allclose(every[0, 4], W[4].reshape(SHAPE), rtol=1e-6)

    with pytest.raises(ValueError):
        estimator.class_gradient(x, label=len(LABELS))
    with pytest.raises(ValueError):
        estimator.class_gradient(x, label=[1, 2, 3])


def test_gradient_queries_count_in_budget():
    estimator, nut = make_estimator(max_evaluations=3)
    x = images(1)
    for label in (0, 1, 2):
        estimator.class_gradient(x, label=label)
    with pytest.raises(BudgetExhausted):
        estimator.class_gradient(x, label=3)
    assert len(nut.gradient_calls) == 3
    assert nut.stc.get_gradient_evaluations() == 3
    assert nut.stc.get_evaluated_individuals() == 3


def test_gradient_cache_hits_are_free():
    estimator, nut = make_estimator(max_evaluations=2)
    x = images(1)
    estimator.loss_gradient(x, np.array([1]))
    estimator.tracker.after_query = MagicMock(wraps=estimator.tracker.after_query)
    again = estimator.loss_gradient(x.copy(), np.array([1]))
    assert len(nut.gradient_calls) == 1
    assert nut.stc.get_gradient_evaluations() == 1
    assert estimator.tracker.after_query.call_count == 0  # a gradient is not a prediction
    np.testing.assert_allclose(again[0], analytic_loss_gradient(x[0], 1), rtol=1e-4, atol=1e-7)

    estimator.class_gradient(x, label=1)  # same image and label, other kind: a new query
    assert len(nut.gradient_calls) == 2
    with pytest.raises(BudgetExhausted):
        estimator.class_gradient(x + 0.25, label=1)  # float32 key: no rounding to uint8


@pytest.mark.parametrize("bad, expected", [
    (np.zeros((3, 2, 3), dtype=np.float32), r"\(3, 2, 3\)"),
    (np.full(SHAPE, np.nan, dtype=np.float32), "non-finite"),
])
def test_invalid_gradient_raises_helpful_error(bad, expected):
    estimator, _ = make_estimator(bad_gradient=bad)
    with pytest.raises(InvalidGradientError, match=expected) as info:
        estimator.class_gradient(images(1), label=1)
    message = str(info.value)
    assert "dog" in message
    assert str(tuple(bad.shape)) in message
    with pytest.raises(InvalidGradientError, match="dog"):
        estimator.loss_gradient(images(1), np.array([1]))


def test_logits_mode_predict_returns_logits():
    estimator, nut = make_estimator(use_logits=True, fitness=lambda result: max(p.value for p in result.predictions))
    x = images(2)

    preds = estimator.predict(x)

    expected = np.stack([logits_of(image) for image in np.clip(np.rint(x), 0, 255)])
    np.testing.assert_allclose(preds, expected, rtol=1e-5)
    top_probabilities = [max(softmax(logits_of(image))) for image in np.clip(np.rint(x), 0, 255)]
    assert estimator.tracker.best_fitness.value == pytest.approx(min(top_probabilities))  # fitness from predictions
    np.testing.assert_allclose(estimator.predict(x[:1]), expected[:1], rtol=1e-5)  # cache hit keeps logits
    assert len(nut.received) == 2


def test_probability_mode_predict_ignores_logits():
    estimator, _ = make_estimator()
    preds = estimator.predict(images(1))
    assert preds.min() >= 0.0
    assert preds.sum() == pytest.approx(1.0)


def test_logits_mode_without_logits_raises_capability_error():
    estimator, _ = make_estimator(use_logits=True, with_logits=False)
    with pytest.raises(NutCapabilityError, match="logits"):
        estimator.predict(images(1))


def test_to_logit_vector_validates_length():
    predictions = [{"label": "cat", "score": 0.6}, {"label": "dog", "score": 0.3}]
    with pytest.raises(NutCapabilityError, match="logits"):
        to_logit_vector(NutRequest({"predictions": predictions}), LABELS)
    with pytest.raises(NutCapabilityError, match="6"):
        to_logit_vector(NutRequest({"predictions": predictions, "logits": [0.1, 0.2]}), LABELS)
    vector = to_logit_vector(NutRequest({"predictions": predictions, "logits": list(range(6))}), LABELS)
    np.testing.assert_array_equal(vector, np.arange(6, dtype=float))


@pytest.mark.parametrize("attack", [FastGradientMethod, ProjectedGradientDescent, CarliniL2Method])
def test_estimator_satisfies_art_requirements(attack):
    estimator, _ = make_estimator(use_logits=True)
    for requirement in attack._estimator_requirements:
        assert isinstance(estimator, requirement)
    assert attack(estimator).estimator is estimator  # ART's own requirement check accepts it


@pytest.mark.parametrize("make_attack", [
    lambda est: FastGradientMethod(est, eps=8.0),
    lambda est: ProjectedGradientDescent(est, eps=8.0, eps_step=2.0, max_iter=3, verbose=False),
    lambda est: CarliniL2Method(est, max_iter=2, binary_search_steps=1, verbose=False),
])
def test_art_white_box_attacks_run_on_the_estimator(make_attack):
    estimator, nut = make_estimator(max_evaluations=1000, use_logits=True)
    x = np.full((1,) + SHAPE, 128.0, dtype=np.float32)
    adversarial = make_attack(estimator).generate(x)
    assert adversarial.shape == x.shape
    assert np.all(np.isfinite(adversarial))
    assert len(nut.gradient_calls) >= 1


def test_cached_gradient_hits_in_a_row_raise_attack_stalled(monkeypatch):
    monkeypatch.setattr(query_tracker, "STALL_LIMIT", 5)
    estimator, nut = make_estimator()
    x = images(1)
    estimator.class_gradient(x, label=1)
    with pytest.raises(AttackStalled):
        for _ in range(5):
            estimator.class_gradient(x, label=1)
    assert len(nut.gradient_calls) == 1


def test_real_gradient_queries_reset_the_stall_counter(monkeypatch):
    monkeypatch.setattr(query_tracker, "STALL_LIMIT", 5)
    estimator, nut = make_estimator()
    estimator.tracker.after_query = MagicMock()
    x = images(1)
    for step in range(4):
        for _ in range(4):  # cached hits, one below the limit
            estimator.predict(np.zeros((1,) + SHAPE, dtype=np.float32))
        estimator.class_gradient(x + step, label=1)  # a real gradient query: NUT activity
    assert len(nut.gradient_calls) == 4
    estimator.tracker.after_query.assert_called_once()  # a gradient never updates the best
