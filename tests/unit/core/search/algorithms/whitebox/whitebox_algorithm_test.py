import logging

import numpy as np
import pytest

from core.config_parser import ConfigParser
from core.search.algorithms.baseline.nut_interface import NutCapabilityError
from core.search.algorithms.whitebox.whitebox_algorithm import WhiteboxAlgorithm
from core.search.phase_controller import PhaseController
from core.search.service.archive import Archive
from core.search.service.fitness_function.targeted_fitness_function import TargetedFitnessFunction
from core.search.service.fitness_function.untargeted_fitness_function import UntargetedFitnessFunction
from core.search.service.randomness import Randomness
from core.search.service.search_time_controller import SearchTimeController
from core.utils.images import ProcessedImage
from core.utils.nut_request import NutRequest

LABELS = ["cat", "dog", "bird", "fish", "frog", "deer"]
# Logits of the original image: 'cat' leads by 2, an L-inf perturbation of 8 can move the margins by about 7.
ORIGINAL_LOGITS = np.array([2.0, 0.0, -0.2, -0.4, -0.6, -0.8])
SMALL = (16, 16)


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


class LinearModel:

    """Linear classifier on 0-255 images: logits = W @ x.ravel() + b, with analytic gradients."""

    def __init__(self, shape, seed=0):
        rng = np.random.default_rng(seed)
        n = int(np.prod(shape))
        self.shape = shape
        self.image = rng.integers(60, 197, size=shape).astype(np.uint8)
        # Scaled so that an L-inf perturbation of 8 changes a logit margin by about 7, whatever the image size.
        self.W = rng.normal(scale=0.77 / n, size=(len(LABELS), n))
        self.b = ORIGINAL_LOGITS - self.W @ self.image.astype(np.float64).ravel()

    def logits(self, image):
        return self.W @ np.asarray(image, dtype=np.float64).ravel() + self.b

    def response(self, image, with_logits=True):
        logits = self.logits(image)
        probs = softmax(logits)
        top5 = np.argsort(-probs)[:5]
        result = {"predictions": [{"label": LABELS[i], "score": float(probs[i])} for i in top5]}
        if with_logits:
            result["logits"] = [float(v) for v in logits]
        return result

    def label(self, image):
        return LABELS[int(np.argmax(self.logits(image)))]

    def gradient(self, image, label, kind):
        index = LABELS.index(label)
        if kind == "class":
            return self.W[index].reshape(self.shape)
        probs = softmax(self.logits(image))
        probs[index] -= 1.0
        return (probs @ self.W).reshape(self.shape)


class LinearNut:

    """Fake remote controller of a white-box NUT: counts every prediction and gradient query."""

    def __init__(self, stc, model, target=None, with_logits=True, nut_info=None):
        self.stc = stc
        self.model = model
        self.target = target
        self.with_logits = with_logits
        self.images = []
        self.gradient_calls = []
        self.nut_info = nut_info if nut_info is not None else {"labels": LABELS,
                                                               "capabilities": ["scores", "gradients"]}

    def new_action(self, image):
        self.stc.new_individual_evaluation()
        self.images.append(np.array(image, copy=True))
        return NutRequest(self.model.response(image, self.with_logits), self.target)

    def gradient(self, image, label, kind):
        self.stc.new_individual_evaluation()
        self.stc.new_gradient_evaluation()
        self.gradient_calls.append((np.array(image, copy=True), label, kind))
        return self.model.gradient(image, label, kind).astype(np.float32)

    def get_nut_info(self):
        return self.nut_info

    @property
    def total_queries(self):
        return len(self.images) + len(self.gradient_calls)


def make_algorithm(attack="pgd", max_evaluations=200, shape=SMALL, targeted=False, target=None, with_logits=True,
                   nut_info=None, **overrides):
    config = dict(ConfigParser().default_params())
    config.update({"algorithm": ConfigParser.Algorithms.WHITEBOX, "whitebox_attack": attack,
                   "max_evaluations": max_evaluations, "seed": 1,
                   "attack_type": ConfigParser.AttackType.TARGETED if targeted else ConfigParser.AttackType.UNTARGETED,
                   "target": target})
    config.update(overrides)
    stc = SearchTimeController(config, pc=PhaseController())
    randomness = Randomness(config)
    archive = Archive(stc, randomness, config)
    model = LinearModel(shape + (3,))
    archive.set_image(ProcessedImage(None, None, model.image))
    nut = LinearNut(stc, model, target=target, with_logits=with_logits, nut_info=nut_info)
    archive.set_original_prediction_results(model.response(model.image, with_logits))
    if targeted:
        ff = TargetedFitnessFunction(archive, nut, stc, target)
    else:
        ff = UntargetedFitnessFunction(archive, nut, stc)
    algorithm = WhiteboxAlgorithm(ff=ff, randomness=randomness, stc=stc, archive=archive, config=config,
                                  mutator=None, crossover=None, sampler=None, apc=None)
    return algorithm, nut


def solution_image(algorithm, solution):
    return algorithm.archive.get_mutated_image(solution.actions)


def linf(algorithm, image):
    return float(np.max(np.abs(image.astype(float) - algorithm.archive.image.array.astype(float))))


@pytest.fixture
def art():
    return pytest.importorskip("art")


def install_fake_attack(monkeypatch, name, generate):
    from core.search.algorithms.whitebox import attacks

    spec = attacks.WHITEBOX_ATTACKS[name]

    class FakeAttack:
        def __init__(self, estimator):
            self.estimator = estimator

        def generate(self, x, y=None, **kwargs):
            return generate(self.estimator, x, y)

    monkeypatch.setitem(attacks.WHITEBOX_ATTACKS, name, attacks.AttackSpec(
        spec.name, spec.requirement, spec.stop_at_success_default, spec.norm_fn,
        lambda estimator, config, targeted: FakeAttack(estimator), spec.gradient_calls_fn))


def test_get_type():
    algorithm, _ = make_algorithm()
    assert algorithm.get_type() == "whitebox"


def test_whitebox_attacks_and_norms():
    from core.search.algorithms.whitebox.attacks import WHITEBOX_ATTACKS
    from core.search.algorithms.baseline.nut_interface import NutRequirement

    assert sorted(WHITEBOX_ATTACKS) == ["cw", "fgsm", "pgd"]
    assert all(spec.requirement == NutRequirement.GRADIENTS for spec in WHITEBOX_ATTACKS.values())
    config = dict(ConfigParser().default_params())
    assert WHITEBOX_ATTACKS["fgsm"].norm(config) == np.inf
    assert WHITEBOX_ATTACKS["pgd"].norm(config) == np.inf
    assert WHITEBOX_ATTACKS["pgd"].norm(dict(config, pgd_norm="2")) == 2
    assert WHITEBOX_ATTACKS["cw"].norm(config) == 2


def test_default_parameters():
    params = ConfigParser().default_params()
    assert params["whitebox_attack"] == "pgd"
    assert params["fgsm_eps"] == 8.0
    assert params["pgd_norm"] == "inf"
    assert params["pgd_eps"] == 8.0
    assert params["pgd_eps_step"] == 2.0
    assert params["cw_confidence"] == 0.0
    for name in ("pgd_max_iter", "cw_learning_rate", "cw_max_iter", "cw_binary_search_steps", "cw_initial_const",
                 "cw_max_halving", "cw_max_doubling"):
        assert params[name] > 0


def test_fgsm_flips_linear_model(art):
    algorithm, nut = make_algorithm("fgsm")
    solution = algorithm.search()
    image = solution_image(algorithm, solution)
    assert algorithm.stc.is_solution(solution.fitness_value)
    assert nut.model.label(image) != "cat"
    assert linf(algorithm, image) <= 8.0 + 0.5
    assert len(nut.gradient_calls) == 1


def test_pgd_flips_linear_model_and_respects_eps(art):
    algorithm, nut = make_algorithm("pgd")
    solution = algorithm.search()
    image = solution_image(algorithm, solution)
    assert algorithm.stc.is_solution(solution.fitness_value)
    assert nut.model.label(image) != "cat"
    assert linf(algorithm, image) <= algorithm.config["pgd_eps"] + 0.5
    assert all(kind == "loss" for _, _, kind in nut.gradient_calls)


def test_pgd_l2_respects_eps(art):
    algorithm, _ = make_algorithm("pgd", pgd_norm="2", pgd_eps=40.0, pgd_eps_step=10.0, pgd_max_iter=20)
    solution = algorithm.search()
    diff = solution_image(algorithm, solution).astype(float) - algorithm.archive.image.array.astype(float)
    # Rounding to uint8 adds at most 0.5 per channel.
    assert np.linalg.norm(diff.ravel()) <= 40.0 + 0.5 * np.sqrt(diff.size)
    assert algorithm.tracker.norm == 2


def test_cw_flips_linear_model(art):
    algorithm, nut = make_algorithm("cw", max_evaluations=1000)
    solution = algorithm.search()
    image = solution_image(algorithm, solution)
    assert algorithm.stc.is_solution(solution.fitness_value)
    assert nut.model.label(image) != "cat"
    assert any(kind == "class" for _, _, kind in nut.gradient_calls)


@pytest.mark.parametrize("attack", ["fgsm", "pgd", "cw"])
@pytest.mark.parametrize("max_evaluations", [3, 15, 60])
def test_budget_counts_gradient_and_prediction_queries(art, attack, max_evaluations):
    algorithm, nut = make_algorithm(attack, max_evaluations=max_evaluations, pgd_norm="2", pgd_eps=200.0,
                                    pgd_eps_step=5.0)
    algorithm.search()
    stc = algorithm.stc
    assert nut.total_queries <= max_evaluations
    assert stc.get_evaluated_individuals() == nut.total_queries
    assert stc.get_gradient_evaluations() == len(nut.gradient_calls) > 0


def test_final_image_is_verified_with_uint8_prediction(art, monkeypatch):
    produced = {}

    def generate(estimator, x, y):
        gradient = estimator.loss_gradient(x, y)
        produced["image"] = np.clip(x + 9.6 * np.sign(gradient), 0, 255)
        return produced["image"]

    install_fake_attack(monkeypatch, "pgd", generate)
    algorithm, nut = make_algorithm("pgd")
    solution = algorithm.search()
    expected = np.clip(np.rint(produced["image"][0]), 0, 255).astype(np.uint8)
    assert len(nut.images) == 1
    assert nut.images[-1].dtype == np.uint8
    assert np.array_equal(nut.images[-1], expected)
    assert algorithm.stc.get_evaluated_individuals() == 2
    assert algorithm.stc.is_solution(solution.fitness_value)
    assert np.array_equal(solution_image(algorithm, solution), expected)
    assert algorithm.stc.get_current_fitness() is algorithm.tracker.best_fitness


def test_budget_exhaustion_verifies_last_attack_image_with_reserved_query(art, monkeypatch):
    def generate(estimator, x, y):
        image = x.copy()
        while True:  # never returns: stopped by the budget
            image = np.clip(image + 1.2 * np.sign(estimator.loss_gradient(image, y)), 0, 255)

    install_fake_attack(monkeypatch, "pgd", generate)
    algorithm, nut = make_algorithm("pgd", max_evaluations=10)
    algorithm.search()
    assert len(nut.gradient_calls) == 9  # one query is kept for the final check
    assert len(nut.images) == 1
    last_requested = nut.gradient_calls[-1][0]
    step = np.clip(last_requested + 1.2 * np.sign(nut.model.gradient(last_requested, "cat", "loss")), 0, 255)
    assert np.array_equal(nut.images[0], np.clip(np.rint(step), 0, 255).astype(np.uint8))
    assert algorithm.stc.get_evaluated_individuals() == 10
    assert algorithm.attack_completed is False


def test_attack_without_queries_spends_the_reserved_query_on_the_final_check(art, monkeypatch):
    install_fake_attack(monkeypatch, "fgsm", lambda estimator, x, y: x + 20.0)
    algorithm, nut = make_algorithm("fgsm", max_evaluations=1)
    algorithm.search()
    assert nut.total_queries == 1
    assert algorithm.attack_completed is True


def test_final_check_is_skipped_when_budget_is_already_used(art, monkeypatch):
    install_fake_attack(monkeypatch, "fgsm", lambda estimator, x, y: x + 20.0)
    algorithm, nut = make_algorithm("fgsm", max_evaluations=1)
    algorithm.stc.evaluated_individuals = 1
    algorithm.search()
    assert nut.total_queries == 0


def test_targeted_pgd_reaches_target(art):
    algorithm, nut = make_algorithm("pgd", targeted=True, target="frog")
    solution = algorithm.search()
    image = solution_image(algorithm, solution)
    assert algorithm.stc.is_solution(solution.fitness_value)
    assert nut.model.label(image) == "frog"


def test_targeted_mode_passes_target_one_hot(art, monkeypatch):
    captured = {}

    def generate(estimator, x, y):
        captured["y"] = y
        return x

    install_fake_attack(monkeypatch, "cw", generate)
    algorithm, _ = make_algorithm("cw", targeted=True, target="fish")
    algorithm.search()
    np.testing.assert_array_equal(captured["y"], [[0, 0, 0, 1, 0, 0]])
    algorithm, _ = make_algorithm("cw")
    algorithm.search()
    np.testing.assert_array_equal(captured["y"], [[1, 0, 0, 0, 0, 0]])


def test_whitebox_without_gradients_capability_raises(art):
    algorithm, nut = make_algorithm("pgd", nut_info={"labels": LABELS, "capabilities": ["scores"]})
    with pytest.raises(NutCapabilityError, match="gradients"):
        algorithm.search()
    assert nut.total_queries == 0
    assert algorithm.stc.get_evaluated_individuals() == 0


def test_whitebox_without_logits_raises_before_any_query(art):
    algorithm, nut = make_algorithm("pgd", with_logits=False)
    with pytest.raises(NutCapabilityError, match="logits"):
        algorithm.search()
    assert nut.images == []
    assert nut.gradient_calls == []
    assert algorithm.stc.get_evaluated_individuals() == 0


@pytest.mark.parametrize("attack", ["fgsm", "pgd", "cw"])
def test_whitebox_rejects_max_pixels_before_any_query(attack):
    algorithm, nut = make_algorithm(attack, max_pixels=5)
    with pytest.raises(ValueError, match="max_pixels"):
        algorithm.search()
    assert nut.total_queries == 0


def test_unknown_whitebox_attack_raises():
    algorithm, nut = make_algorithm("deepfool")
    with pytest.raises(ValueError, match="whitebox_attack"):
        algorithm.search()
    assert nut.total_queries == 0


def test_warns_when_iterations_exceed_budget(art, caplog):
    algorithm, _ = make_algorithm("pgd", max_evaluations=20, pgd_max_iter=50)
    with caplog.at_level(logging.WARNING):
        algorithm.search()
    assert "pgd_max_iter" in caplog.text

    caplog.clear()
    algorithm, _ = make_algorithm("pgd", max_evaluations=200, pgd_max_iter=50)
    with caplog.at_level(logging.WARNING):
        algorithm.search()
    assert "pgd_max_iter" not in caplog.text


def test_logs_art_version(art, caplog):
    algorithm, _ = make_algorithm("fgsm")
    with caplog.at_level(logging.INFO, logger="core.search.algorithms.whitebox.whitebox_algorithm"):
        algorithm.search()
    assert "ART %s" % art.__version__ in caplog.text


class CountingCalls:

    """Wraps an estimator method and counts the calls that returned (were not stopped by the budget)."""

    def __init__(self, method):
        self.method = method
        self.completed = 0

    def __call__(self, *args, **kwargs):
        result = self.method(*args, **kwargs)
        self.completed += 1
        return result


@pytest.mark.parametrize("attack", ["fgsm", "pgd", "cw"])
def test_defaults_fit_budget_on_224_images(art, monkeypatch, caplog, attack):
    from core.search.algorithms.baseline import estimator as estimator_module

    counters = {}
    original_init = estimator_module.RemoteNutEstimator.__init__

    def counting_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        counters["loss"] = self.loss_gradient = CountingCalls(self.loss_gradient)
        counters["class"] = self.class_gradient = CountingCalls(self.class_gradient)

    monkeypatch.setattr(estimator_module.RemoteNutEstimator, "__init__", counting_init)
    algorithm, nut = make_algorithm(attack, max_evaluations=1000, shape=(224, 224))
    with caplog.at_level(logging.WARNING):
        algorithm.search()
    config = algorithm.config
    assert nut.total_queries <= 1000
    if attack == "fgsm":
        # One loss gradient for the single step, then the final image is checked with a prediction.
        assert algorithm.attack_completed is True
        assert counters["loss"].completed == 1 and counters["class"].completed == 0
        assert nut.total_queries <= 5
    elif attack == "pgd":
        assert algorithm.attack_completed is True
        assert counters["loss"].completed == config["pgd_max_iter"]
        assert "pgd_max_iter" not in caplog.text
    else:
        # A binary search step runs cw_max_iter iterations with two class gradients each.
        assert counters["class"].completed >= 2 * config["cw_max_iter"]


def test_application_wires_whitebox_algorithm():
    from core.problem.base_module import BaseModule
    from core.utils.application import configure_container

    container = BaseModule()
    config = dict(ConfigParser().default_params())
    config["algorithm"] = ConfigParser.Algorithms.WHITEBOX
    container.config.override(config)
    container = configure_container(container)
    algorithm = container.algorithm()
    assert isinstance(algorithm, WhiteboxAlgorithm)
    assert algorithm.ff is container.ff()
    assert algorithm.archive is container.archive()


def test_shared_base_class():
    from core.search.algorithms.baseline.attack_runner import AttackRunnerAlgorithm
    from core.search.algorithms.baseline.baseline_algorithm import BaselineAlgorithm

    assert issubclass(WhiteboxAlgorithm, AttackRunnerAlgorithm)
    assert issubclass(BaselineAlgorithm, AttackRunnerAlgorithm)
