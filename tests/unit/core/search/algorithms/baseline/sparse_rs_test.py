import os
import sys

import numpy as np
import pytest

from core.search.algorithms.baseline.attacks import ATTACKS, resolve_sparse_rs_k
from core.search.algorithms.baseline.nut_interface import NutRequirement
from core.search.algorithms.baseline.sparse_rs import SparseRS
from core.utils.nut_request import NutRequest
from core.utils.perturbation import l0_pixels
from tests.unit.core.search.algorithms.baseline.baseline_algorithm_test import (
    HEIGHT, WIDTH, constant_scores, make_algorithm, make_image, solution_image)

P_INIT = 0.3


@pytest.fixture
def art():
    return pytest.importorskip("art")


def changed_pixels(original, image):
    return np.any(np.asarray(image) != np.asarray(original), axis=-1)


def test_sparse_rs_is_registered_as_score_based_l0_attack():
    spec = ATTACKS["sparse_rs"]
    assert spec.name == "sparse_rs"
    assert spec.requirement == NutRequirement.SCORES
    assert spec.stop_at_success_default is True
    assert spec.norm({}) == 0


def test_sparse_rs_finds_adversarial_with_k_pixels(art):
    algorithm, nut = make_algorithm("sparse_rs", max_evaluations=300, image=make_image(mean=124), sparse_rs_k=40)
    solution = algorithm.search()
    stc = algorithm.stc
    assert stc.is_solution(solution.fitness_value)
    assert stc.get_first_success_eval() == stc.get_evaluated_individuals() < 300
    image = solution_image(algorithm, solution)
    assert np.array_equal(image, algorithm.tracker.best_image)
    assert l0_pixels(algorithm.archive.image.array, image) <= 40


def test_sparse_rs_colors_are_corners(art):
    algorithm, nut = make_algorithm("sparse_rs", max_evaluations=300, sparse_rs_k=40)
    solution = algorithm.search()
    original = algorithm.archive.image.array
    best = solution_image(algorithm, solution)
    mask = changed_pixels(original, best)
    assert mask.any()
    assert np.isin(best[mask], [0, 255]).all()
    for image in nut.images:
        assert np.isin(image[changed_pixels(original, image)], [0, 255]).all()


def test_sparse_rs_budget_is_never_exceeded(art):
    algorithm, nut = make_algorithm("sparse_rs", max_evaluations=50, scores_fn=constant_scores)
    solution = algorithm.search()
    assert len(nut.images) == 50
    assert algorithm.stc.get_evaluated_individuals() == 50
    assert algorithm.stc.get_first_success_eval() is None
    assert not algorithm.stc.is_solution(solution.fitness_value)


def test_sparse_rs_sends_only_new_images_with_exactly_k_pixels(art):
    algorithm, nut = make_algorithm("sparse_rs", max_evaluations=120, scores_fn=constant_scores, sparse_rs_k=6)
    algorithm.search()
    original = algorithm.archive.image.array
    assert len(nut.images) == 120
    assert len({img.tobytes() for img in nut.images}) == 120
    assert all(l0_pixels(original, img) == 6 for img in nut.images)


def test_sparse_rs_same_seed_same_result(art):
    results = []
    for _ in range(2):
        algorithm, nut = make_algorithm("sparse_rs", max_evaluations=200, seed=7, sparse_rs_k=40)
        solution = algorithm.search()
        results.append((nut.images, solution_image(algorithm, solution)))
    assert len(results[0][0]) == len(results[1][0])
    assert all(np.array_equal(a, b) for a, b in zip(results[0][0], results[1][0]))
    assert np.array_equal(results[0][1], results[1][1])

    algorithm, nut = make_algorithm("sparse_rs", max_evaluations=200, seed=8, sparse_rs_k=40)
    algorithm.search()
    assert not np.array_equal(nut.images[0], results[0][0][0])


def test_sparse_rs_targeted_reaches_target(art):
    algorithm, nut = make_algorithm("sparse_rs", max_evaluations=300, targeted=True, target="bright",
                                    sparse_rs_k=40)
    solution = algorithm.search()
    assert algorithm.stc.is_solution(solution.fitness_value)
    predictions = NutRequest(nut.response(solution_image(algorithm, solution)))
    assert predictions.max_score.label == "bright"


@pytest.mark.parametrize("sparse_rs_k, max_pixels, expected", [(0, 0, 10), (0, 7, 7), (4, 7, 4), (5, 0, 5)])
def test_sparse_rs_k_defaults_to_max_pixels_then_10(art, sparse_rs_k, max_pixels, expected):
    assert resolve_sparse_rs_k({"sparse_rs_k": sparse_rs_k, "max_pixels": max_pixels}) == expected
    algorithm, nut = make_algorithm("sparse_rs", max_evaluations=5, scores_fn=constant_scores,
                                    sparse_rs_k=sparse_rs_k, max_pixels=max_pixels)
    algorithm.search()
    original = algorithm.archive.image.array
    assert len(nut.images) == 5
    assert all(l0_pixels(original, img) == expected for img in nut.images)


def test_sparse_rs_k_above_max_pixels_raises_before_any_query():
    algorithm, nut = make_algorithm("sparse_rs", sparse_rs_k=8, max_pixels=7)
    with pytest.raises(ValueError, match="sparse_rs_k"):
        algorithm.search()
    assert nut.images == []


def test_sparse_rs_k_must_leave_unchanged_pixels(art):
    algorithm, nut = make_algorithm("sparse_rs", sparse_rs_k=HEIGHT * WIDTH)
    with pytest.raises(ValueError, match="sparse_rs_k"):
        algorithm.search()
    assert nut.images == []


def test_p_selection_schedule_matches_official():
    attack = SparseRS(None, k=10, p_init=P_INIT, max_iter=10000, targeted=False, randomness_seed=0)
    expected = {0: P_INIT, 11: P_INIT / 2, 51: P_INIT / 4, 201: P_INIT / 5, 501: P_INIT / 6, 1001: P_INIT / 8,
                2001: P_INIT / 10, 4001: P_INIT / 12, 6001: P_INIT / 15, 8001: P_INIT / 20}
    for it, p in expected.items():
        assert attack.p_selection(it) == pytest.approx(p), it
    # Boundaries are inclusive on the right, as in the official code.
    assert attack.p_selection(50) == pytest.approx(P_INIT / 2)
    assert attack.p_selection(8000) == pytest.approx(P_INIT / 15)


def test_p_selection_is_rescaled_to_max_iter():
    attack = SparseRS(None, k=10, p_init=P_INIT, max_iter=1000, targeted=False, randomness_seed=0)
    assert attack.p_selection(5) == pytest.approx(P_INIT / 2)       # 50 on the 10000 scale
    assert attack.p_selection(100) == pytest.approx(P_INIT / 6)     # 1000
    assert attack.p_selection(101) == pytest.approx(P_INIT / 8)     # 1010
    assert attack.p_selection(900) == pytest.approx(P_INIT / 20)    # 9000


class _StopAtSuccess(Exception):
    pass


class _NumpyLinearEstimator:

    """Softmax of a linear model on HxWxC images (0-255); counts predictions, stops at the first misclassification."""

    def __init__(self, weights, bias, label):
        self.weights, self.bias, self.label = weights, bias, label
        self.calls = 0

    def predict(self, x):
        self.calls += 1
        logits = (np.asarray(x, dtype=np.float64).reshape(len(x), -1) / 255.0) @ self.weights.T + self.bias
        if int(np.argmax(logits[0])) != self.label:
            raise _StopAtSuccess()
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


N_CLASSES, SIDE, OFFICIAL_K, OFFICIAL_QUERIES = 10, 32, 10, 200


def _linear_problem(seed):
    """Random linear model and image whose original class 0 leads by a margin that 10 pixels can overcome.

    With margin 24, 200 queries and k=10 both implementations succeed on about 90-100% of seeds 0-19 with a median
    of about 110 queries; on 300 seeds with margin 26 they gave 0.81 / median 129 (official) vs 0.80 / 124 (port).
    """
    rng = np.random.RandomState(1000 + seed)
    weights = rng.normal(size=(N_CLASSES, SIDE * SIDE * 3))
    image = rng.randint(0, 256, size=(SIDE, SIDE, 3)).astype(np.float64)
    logits = weights @ (image.ravel() / 255.0)
    bias = -logits
    bias[0] += 24.0
    return weights, bias, image


def _run_ours(seed):
    weights, bias, image = _linear_problem(seed)
    estimator = _NumpyLinearEstimator(weights, bias, 0)
    y = np.zeros((1, N_CLASSES), dtype=np.float32)
    y[0, 0] = 1
    attack = SparseRS(estimator, k=OFFICIAL_K, p_init=P_INIT, max_iter=OFFICIAL_QUERIES, targeted=False,
                      randomness_seed=seed)
    try:
        attack.generate(image[np.newaxis].astype(np.float32), y)
    except _StopAtSuccess:
        return True, estimator.calls
    return False, estimator.calls


def _run_official(rs_attack_class, torch, seed):
    weights, bias, image = _linear_problem(seed)
    w = torch.tensor(weights, dtype=torch.float64)
    b = torch.tensor(bias, dtype=torch.float64)

    def model(x):
        return x.double().permute(0, 2, 3, 1).reshape(x.shape[0], -1) @ w.T + b

    x = torch.tensor(image / 255.0).permute(2, 0, 1).unsqueeze(0).float()
    attack = rs_attack_class(model, norm="L0", eps=OFFICIAL_K, n_queries=OFFICIAL_QUERIES, p_init=P_INIT,
                             seed=seed, verbose=False, loss="margin")
    queries, adv = attack.perturb(x, torch.tensor([0]))
    success = int(model(adv).argmax(1)[0]) != 0
    return success, float(queries[0])


def test_sparse_rs_matches_official_distribution():
    checkout = os.environ.get("SPARSE_RS_OFFICIAL")
    if not checkout or not os.path.isfile(os.path.join(checkout, "rs_attacks.py")):
        pytest.skip("Set SPARSE_RS_OFFICIAL to a checkout of github.com/fra31/sparse-rs to run this comparison.")
    torch = pytest.importorskip("torch")
    sys.path.insert(0, checkout)
    try:
        from rs_attacks import RSAttack  # type: ignore[import-not-found]
    finally:
        sys.path.remove(checkout)

    ours = [_run_ours(seed) for seed in range(20)]
    official = [_run_official(RSAttack, torch, seed) for seed in range(20)]
    ours_rate = np.mean([s for s, _ in ours])
    official_rate = np.mean([s for s, _ in official])
    assert official_rate > 0.0, "the toy problem must not be impossible"
    assert abs(ours_rate - official_rate) <= 0.2 + 1e-9
    ours_median = np.median([q for s, q in ours if s])
    official_median = np.median([q for s, q in official if s])
    assert official_median > 20, "the toy problem must not be trivial"
    assert abs(ours_median - official_median) <= 0.3 * official_median
