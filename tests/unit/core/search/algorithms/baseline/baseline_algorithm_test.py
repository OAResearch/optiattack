import logging
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.config_parser import ConfigParser
from core.search.action import Action
from core.search.algorithms.baseline.baseline_algorithm import BaselineAlgorithm
from core.search.algorithms.baseline.nut_interface import NutCapabilityError
from core.search.algorithms.baseline.query_tracker import STALL_LIMIT
from core.search.fitness_value import FitnessValue
from core.search.phase_controller import PhaseController
from core.search.service.archive import Archive
from core.search.service.fitness_function.targeted_fitness_function import TargetedFitnessFunction
from core.search.service.fitness_function.untargeted_fitness_function import UntargetedFitnessFunction
from core.search.service.randomness import Randomness
from core.search.service.search_time_controller import SearchTimeController
from core.utils.images import ProcessedImage
from core.utils.nut_request import NutRequest

LABELS = ["dark", "bright", "mid"]
HEIGHT, WIDTH = 16, 24
THRESHOLD = 128.0
ATTACK_NAMES = ["square", "hsj", "simba", "pixel"]


def brightness_scores(image):
    """Scores that only depend on the mean brightness: above THRESHOLD the image becomes 'bright'."""
    mean = float(np.mean(image))
    bright = 1.0 / (1.0 + np.exp(-(mean - THRESHOLD) / 2.0))
    return {"dark": 0.7 * (1.0 - bright), "bright": bright, "mid": 0.3 * (1.0 - bright)}


def constant_scores(image):
    """Scores of a NUT that can never be fooled."""
    return {"dark": 0.8, "bright": 0.15, "mid": 0.05}


class FakeNut:

    """Fake remote controller: counts every query and records the images it receives."""

    def __init__(self, stc, scores_fn=brightness_scores, target=None, nut_info=None):
        self.stc = stc
        self.scores_fn = scores_fn
        self.target = target
        self.images = []
        self.nut_info = nut_info if nut_info is not None else {"labels": LABELS, "capabilities": ["scores"]}

    def response(self, image):
        scores = self.scores_fn(image)
        return {"predictions": [{"label": label, "score": float(scores[label])} for label in LABELS]}

    def new_action(self, image):
        self.stc.new_individual_evaluation()
        self.images.append(np.array(image, copy=True))
        return NutRequest(self.response(image), self.target)

    def get_nut_info(self):
        return self.nut_info


def make_image(height=HEIGHT, width=WIDTH, mean=124):
    rng = np.random.RandomState(0)
    image = rng.randint(-20, 21, size=(height, width, 3)) + mean
    return np.clip(image, 0, 255).astype(np.uint8)


def make_algorithm(attack="square", max_evaluations=150, scores_fn=brightness_scores, image=None,
                   targeted=False, target=None, nut_info=None, seed=1, **overrides):
    config = dict(ConfigParser().default_params())
    config.update({"algorithm": ConfigParser.Algorithms.BASELINE, "baseline_attack": attack,
                   "max_evaluations": max_evaluations, "seed": seed,
                   "attack_type": ConfigParser.AttackType.TARGETED if targeted else ConfigParser.AttackType.UNTARGETED,
                   "target": target})
    if attack == "simba":
        # ART's SimBA 'dct' basis only works on square images; the default test image is 16x24.
        config["simba_attack"] = "px"
    config.update(overrides)
    stc = SearchTimeController(config, pc=PhaseController())
    randomness = Randomness(config)
    archive = Archive(stc, randomness, config)
    image = make_image() if image is None else image
    archive.set_image(ProcessedImage(None, None, image))
    nut = FakeNut(stc, scores_fn, target=target, nut_info=nut_info)
    archive.set_original_prediction_results(nut.response(image))
    if targeted:
        ff = TargetedFitnessFunction(archive, nut, stc, target)
    else:
        ff = UntargetedFitnessFunction(archive, nut, stc)
    algorithm = BaselineAlgorithm(ff=ff, randomness=randomness, stc=stc, archive=archive, config=config,
                                  mutator=None, crossover=None, sampler=None, apc=None)
    return algorithm, nut


def solution_image(algorithm, solution):
    return algorithm.archive.get_mutated_image(solution.actions)


@pytest.fixture
def art():
    return pytest.importorskip("art")


def test_get_type():
    algorithm, _ = make_algorithm()
    assert algorithm.get_type() == "baseline"


def test_square_finds_adversarial_and_stops_at_first_success(art):
    algorithm, nut = make_algorithm("square", max_evaluations=200)
    solution = algorithm.search()
    stc = algorithm.stc
    assert stc.get_first_success_eval() is not None
    assert stc.get_first_success_eval() == stc.get_evaluated_individuals()
    assert stc.get_evaluated_individuals() < 200
    assert stc.is_solution(solution.fitness_value)
    assert np.array_equal(solution_image(algorithm, solution), algorithm.tracker.best_image)
    assert np.array_equal(nut.images[-1], algorithm.tracker.best_image)


def test_hsj_uses_whole_budget_and_reports_smallest_successful_perturbation(art):
    algorithm, nut = make_algorithm("hsj", max_evaluations=150)
    solution = algorithm.search()
    stc = algorithm.stc
    assert stc.get_evaluated_individuals() == 150
    first = stc.get_first_success_eval()
    assert first is not None and first < 150

    original = algorithm.archive.image.array.astype(float)
    successful = [img for img in nut.images
                  if stc.is_solution(FitnessValue(algorithm.ff.fitness_from_result(
                      NutRequest(nut.response(img))), []))]
    smallest = min(np.linalg.norm((img.astype(float) - original).ravel()) for img in successful)
    best = algorithm.tracker.best_image
    assert np.linalg.norm((best.astype(float) - original).ravel()) == pytest.approx(smallest)
    assert np.array_equal(solution_image(algorithm, solution), best)
    assert stc.is_solution(solution.fitness_value)


@pytest.mark.parametrize("attack", ATTACK_NAMES)
def test_budget_is_never_exceeded(art, attack):
    algorithm, nut = make_algorithm(attack, max_evaluations=40, scores_fn=constant_scores)
    solution = algorithm.search()
    assert algorithm.stc.get_evaluated_individuals() <= 40
    assert len(nut.images) == algorithm.stc.get_evaluated_individuals()
    assert algorithm.stc.get_first_success_eval() is None
    assert not algorithm.stc.is_solution(solution.fitness_value)


def test_same_seed_gives_same_result(art):
    results = []
    for _ in range(2):
        algorithm, nut = make_algorithm("square", max_evaluations=200, seed=7)
        solution = algorithm.search()
        results.append((algorithm.stc.get_evaluated_individuals(), solution_image(algorithm, solution)))
    assert results[0][0] == results[1][0]
    assert np.array_equal(results[0][1], results[1][1])


def test_targeted_mode_passes_target_to_attack(art, monkeypatch):
    from core.search.algorithms.baseline import attacks

    captured = {}

    class RecordingAttack:
        def generate(self, x, y=None, **kwargs):
            captured["x"], captured["y"] = x, y
            return x

    def fake_build(estimator, config, targeted):
        captured["targeted"] = targeted
        return RecordingAttack()

    spec = attacks.ATTACKS["hsj"]
    monkeypatch.setitem(attacks.ATTACKS, "hsj", attacks.AttackSpec(spec.name, spec.requirement,
                                                                    spec.stop_at_success_default,
                                                                    spec.norm_fn, fake_build))
    algorithm, _ = make_algorithm("hsj", targeted=True, target="bright")
    algorithm.search()
    assert captured["targeted"] is True
    np.testing.assert_array_equal(captured["y"], [[0, 1, 0]])
    assert captured["x"].shape == (1, HEIGHT, WIDTH, 3)

    algorithm, _ = make_algorithm("hsj")
    algorithm.search()
    assert captured["targeted"] is False
    np.testing.assert_array_equal(captured["y"], [[1, 0, 0]])


def test_targeted_square_reaches_the_target(art):
    algorithm, nut = make_algorithm("square", max_evaluations=200, targeted=True, target="bright")
    solution = algorithm.search()
    assert algorithm.stc.is_solution(solution.fitness_value)
    predictions = NutRequest(nut.response(solution_image(algorithm, solution)))
    assert predictions.max_score.label == "bright"


def test_unknown_target_raises_helpful_error(art):
    algorithm, _ = make_algorithm("square", targeted=True, target="zebra")
    with pytest.raises(ValueError, match="zebra"):
        algorithm.search()


def test_baseline_without_labels_raises_helpful_error(art):
    algorithm, _ = make_algorithm("square", nut_info={"capabilities": ["scores"]})
    with pytest.raises(NutCapabilityError, match="labels="):
        algorithm.search()


def test_missing_art_raises_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "art", None)
    monkeypatch.delitem(sys.modules, "core.search.algorithms.baseline.estimator", raising=False)
    algorithm, _ = make_algorithm("square")
    with pytest.raises(RuntimeError, match="pip install -r requirements-baselines.txt"):
        algorithm.search()


@pytest.mark.parametrize("attack", ATTACK_NAMES)
@pytest.mark.parametrize("shape", [(16, 24), (24, 16)])
def test_baseline_runs_on_non_square_image(art, attack, shape):
    algorithm, nut = make_algorithm(attack, max_evaluations=200, image=make_image(*shape))
    solution = algorithm.search()
    assert algorithm.stc.get_evaluated_individuals() <= 200
    assert all(img.shape == shape + (3,) for img in nut.images)
    for action in solution.actions:
        row, col = action.get_location()
        assert 0 <= row < shape[0] and 0 <= col < shape[1]
    assert np.array_equal(solution_image(algorithm, solution), algorithm.tracker.best_image)


def test_simba_dct_works_on_square_images(art):
    algorithm, nut = make_algorithm("simba", max_evaluations=200, image=make_image(16, 16, mean=127),
                                    simba_attack="dct")
    solution = algorithm.search()
    assert algorithm.stc.is_solution(solution.fitness_value)
    assert algorithm.stc.get_first_success_eval() == algorithm.stc.get_evaluated_individuals()
    assert np.array_equal(solution_image(algorithm, solution), algorithm.tracker.best_image)


def test_simba_dct_on_non_square_image_raises_helpful_error(art):
    algorithm, _ = make_algorithm("simba", simba_attack="dct")
    with pytest.raises(ValueError, match="square images"):
        algorithm.search()


def test_reverted_pixels_are_not_in_the_solution():
    algorithm, _ = make_algorithm("square")
    original = algorithm.archive.image.array
    first = original.copy()
    first[0, 0] = 255 - first[0, 0]
    first[2, 3] = 255 - first[2, 3]
    algorithm._on_new_best(first, FitnessValue(0.5, []))
    assert {a.get_location() for a in algorithm.archive.extract_solution().actions} == {(0, 0), (2, 3)}

    second = original.copy()
    second[2, 3] = first[2, 3]
    algorithm._on_new_best(second, FitnessValue(0.1, []))
    solution = algorithm.archive.extract_solution()
    assert [a.get_location() for a in solution.actions] == [(2, 3)]
    assert np.array_equal(algorithm.archive.get_mutated_image(solution.actions), second)
    assert algorithm.stc.get_current_fitness_value() == 0.1


@pytest.mark.parametrize("attack, warns", [("simba", True), ("pixel", True), ("square", False), ("hsj", True)])
def test_warns_about_threshold_for_simba_pixel_and_hsj(art, caplog, attack, warns):
    algorithm, _ = make_algorithm(attack, max_evaluations=5, scores_fn=constant_scores, fitness_threshold=-0.1)
    with caplog.at_level(logging.WARNING):
        algorithm.search()
    assert ("fitness_threshold" in caplog.text) is warns


def test_warns_about_pruning_cost_for_large_solutions(caplog):
    algorithm, _ = make_algorithm("square", enable_pruning=True)
    actions = [Action((r, c), 0, 0, 0) for r in range(HEIGHT) for c in range(WIDTH)]
    algorithm.archive.replace_with(MagicMock(individual=MagicMock(get_actions=lambda: actions * 3),
                                             fitness=FitnessValue(0.5, [])))
    with caplog.at_level(logging.WARNING):
        algorithm._warn_about_pruning_cost()
    assert "pruning" in caplog.text.lower()

    caplog.clear()
    algorithm.archive.replace_with(MagicMock(individual=MagicMock(get_actions=lambda: actions[:10]),
                                             fitness=FitnessValue(0.5, [])))
    with caplog.at_level(logging.WARNING):
        algorithm._warn_about_pruning_cost()
    assert caplog.text == ""


def test_no_pruning_warning_when_pruning_disabled(caplog):
    algorithm, _ = make_algorithm("square", enable_pruning=False)
    actions = [Action((r, c), 0, 0, 0) for r in range(HEIGHT) for c in range(WIDTH)] * 3
    algorithm.archive.replace_with(MagicMock(individual=MagicMock(get_actions=lambda: actions),
                                             fitness=FitnessValue(0.5, [])))
    with caplog.at_level(logging.WARNING):
        algorithm._warn_about_pruning_cost()
    assert caplog.text == ""


def test_application_wires_baseline_algorithm():
    from core.problem.base_module import BaseModule
    from core.utils.application import configure_container

    container = BaseModule()
    config = dict(ConfigParser().default_params())
    config["algorithm"] = ConfigParser.Algorithms.BASELINE
    container.config.override(config)
    container = configure_container(container)
    algorithm = container.algorithm()
    assert isinstance(algorithm, BaselineAlgorithm)
    assert algorithm.ff is container.ff()
    assert algorithm.archive is container.archive()


@pytest.mark.parametrize("norm, shape", [("2", (17, 30)), ("2", (17, 17)), ("inf", (17, 30)), ("inf", (30, 17))])
def test_square_runs_on_images_with_odd_shorter_side(art, norm, shape):
    algorithm, nut = make_algorithm("square", max_evaluations=60, image=make_image(*shape),
                                    scores_fn=constant_scores, square_norm=norm)
    algorithm.search()
    assert algorithm.stc.get_evaluated_individuals() == 60


def test_square_l2_on_tall_image_raises_helpful_error(art):
    algorithm, nut = make_algorithm("square", image=make_image(30, 17), square_norm="2")
    with pytest.raises(ValueError, match="square_norm inf"):
        algorithm.search()
    assert nut.images == []


@pytest.mark.parametrize("targeted", [False, True])
def test_square_only_sends_unique_images_and_budget_is_spent_on_them(art, targeted):
    algorithm, nut = make_algorithm("square", max_evaluations=90, scores_fn=constant_scores,
                                    targeted=targeted, target="mid" if targeted else None)
    algorithm.search()
    assert len(nut.images) == 90
    assert algorithm.stc.get_evaluated_individuals() == 90
    assert len({img.tobytes() for img in nut.images}) == 90


def test_hsj_same_seed_gives_same_result(art):
    results = []
    for _ in range(2):
        algorithm, nut = make_algorithm("hsj", max_evaluations=120, seed=11)
        algorithm.search()
        results.append((algorithm.stc.get_evaluated_individuals(), algorithm.tracker.best_image))
    assert results[0][0] == results[1][0]
    assert np.array_equal(results[0][1], results[1][1])


def test_hsj_receives_found_start_image(art, monkeypatch):
    from core.search.algorithms.baseline import attacks

    captured = {}

    class RecordingAttack:
        def generate(self, x, y=None, **kwargs):
            captured.update(kwargs)
            return x

    spec = attacks.ATTACKS["hsj"]
    monkeypatch.setitem(attacks.ATTACKS, "hsj", attacks.AttackSpec(spec.name, spec.requirement,
                                                                    spec.stop_at_success_default,
                                                                    spec.norm_fn, lambda e, c, t: RecordingAttack()))
    algorithm, nut = make_algorithm("hsj", max_evaluations=100)
    algorithm.search()
    init = captured["x_adv_init"]
    assert init.shape == (1, HEIGHT, WIDTH, 3) and init.dtype == np.float32
    assert algorithm.tracker.best_image is not None


def test_hsj_without_start_image_warns_and_skips_generate(art, caplog, monkeypatch):
    from core.search.algorithms.baseline import attacks

    spec = attacks.ATTACKS["hsj"]

    class FailingAttack:
        def generate(self, x, y=None, **kwargs):
            raise AssertionError("ART must not run without a start image")

    def fail_build(estimator, config, targeted):
        return FailingAttack()

    monkeypatch.setitem(attacks.ATTACKS, "hsj", attacks.AttackSpec(spec.name, spec.requirement,
                                                                    spec.stop_at_success_default,
                                                                    spec.norm_fn, fail_build))
    algorithm, nut = make_algorithm("hsj", max_evaluations=100, scores_fn=constant_scores, hsj_init_size=5)
    with caplog.at_level(logging.WARNING):
        solution = algorithm.search()
    assert "start image" in caplog.text
    assert len(nut.images) == 5
    assert algorithm.stc.get_evaluated_individuals() == 5
    assert solution is not None


@pytest.mark.parametrize("attack, stop, warns", [("simba", "false", True), ("pixel", "false", True),
                                                 ("simba", "auto", False), ("pixel", "true", False),
                                                 ("square", "false", False), ("hsj", "false", False)])
def test_warns_that_stop_at_success_false_has_no_effect_for_simba_and_pixel(art, caplog, attack, stop, warns):
    algorithm, _ = make_algorithm(attack, max_evaluations=5, scores_fn=constant_scores,
                                  baseline_stop_at_success=stop)
    with caplog.at_level(logging.WARNING):
        algorithm.search()
    assert ("baseline_stop_at_success" in caplog.text) is warns


def test_logs_art_version(art, caplog):
    algorithm, _ = make_algorithm("square", max_evaluations=5, scores_fn=constant_scores)
    with caplog.at_level(logging.INFO, logger="core.search.algorithms.baseline.baseline_algorithm"):
        algorithm.search()
    assert "ART %s" % art.__version__ in caplog.text


@pytest.mark.parametrize("attack", ATTACK_NAMES)
def test_original_image_is_not_sent_to_the_nut(art, attack):
    algorithm, nut = make_algorithm(attack, max_evaluations=60, scores_fn=constant_scores)
    algorithm.search()
    original = algorithm.archive.image.array
    assert not any(np.array_equal(img, original) for img in nut.images)
    assert len(nut.images) == algorithm.stc.get_evaluated_individuals()


def test_square_does_not_send_the_original_image_first(art):
    algorithm, nut = make_algorithm("square", max_evaluations=20, scores_fn=constant_scores)
    algorithm.search()
    assert not np.array_equal(nut.images[0], algorithm.archive.image.array)
    assert algorithm.stc.get_evaluated_individuals() == 20


@pytest.mark.parametrize("attack", ["square", "hsj"])
def test_archive_is_built_lazily_and_always_reproduces_the_best_image(art, monkeypatch, attack):
    builds, improvements = [], []
    build = BaselineAlgorithm._individual_from_best
    on_new_best = BaselineAlgorithm._on_new_best

    def counting_build(self, image, fitness):
        builds.append(1)
        return build(self, image, fitness)

    def counting_on_new_best(self, image, fitness):
        improvements.append(1)
        on_new_best(self, image, fitness)

    monkeypatch.setattr(BaselineAlgorithm, "_individual_from_best", counting_build)
    monkeypatch.setattr(BaselineAlgorithm, "_on_new_best", counting_on_new_best)
    algorithm, nut = make_algorithm(attack, max_evaluations=150)
    snapshots = []
    new_action = nut.new_action

    def checking_new_action(image):
        tracker = algorithm.tracker
        if len(nut.images) in (4, 8, 30, 60, 90) and tracker is not None and tracker.best_image is not None:
            solution = algorithm.archive.extract_solution()
            snapshots.append(np.array_equal(algorithm.archive.get_mutated_image(solution.actions),
                                            tracker.best_image))
            assert solution.fitness_value is tracker.best_fitness
        return new_action(image)

    nut.new_action = checking_new_action
    solution = algorithm.search()
    # One build per snapshot plus one for the end of the search, however many improvements there were.
    assert len(builds) <= len(snapshots) + 1 < len(improvements)
    assert snapshots and all(snapshots)
    assert np.array_equal(solution_image(algorithm, solution), algorithm.tracker.best_image)
    assert algorithm.stc.get_current_fitness() is algorithm.tracker.best_fitness


def test_hsj_without_start_image_still_warns_about_pruning_cost(art, caplog, monkeypatch):
    from core.search.algorithms.baseline import baseline_algorithm

    def slightly_improving_scores(image):
        """Scores that improve a little as the image changes but never leave 'dark'."""
        d = 0.01 * min(1.0, abs(float(np.mean(image)) - 124.0) / 255.0)
        return {"dark": 0.8 - d, "bright": 0.15 + d, "mid": 0.05}

    monkeypatch.setattr(baseline_algorithm, "PRUNING_WARNING_ACTIONS", 10)
    algorithm, nut = make_algorithm("hsj", max_evaluations=100, scores_fn=slightly_improving_scores,
                                    hsj_init_size=5, enable_pruning=True)
    with caplog.at_level(logging.WARNING):
        algorithm.search()
    assert "start image" in caplog.text
    assert "pruning" in caplog.text.lower()


def _install_fake_attack(monkeypatch, name, generate):
    from core.search.algorithms.baseline import attacks

    spec = attacks.ATTACKS[name]

    class FakeAttack:
        def __init__(self, estimator):
            self.estimator = estimator

        def generate(self, x, y=None, **kwargs):
            return generate(self.estimator, x)

    monkeypatch.setitem(attacks.ATTACKS, name, attacks.AttackSpec(spec.name, spec.requirement,
                                                                   spec.stop_at_success_default, spec.norm_fn,
                                                                   lambda estimator, config, targeted:
                                                                   FakeAttack(estimator)))


def test_attack_without_queries_reports_original_prediction(art, monkeypatch):
    _install_fake_attack(monkeypatch, "square", lambda estimator, x: x)
    algorithm, nut = make_algorithm("square")
    solution = algorithm.search()
    original = algorithm.archive.get_original_prediction_results()
    assert solution.actions == []
    assert [p.label for p in algorithm.stc.get_current_fitness().predictions] == \
        [p.label for p in original.predictions]
    assert algorithm.stc.get_current_fitness_value() == algorithm.ff.fitness_from_result(original)
    assert not algorithm.stc.is_solution(solution.fitness_value)
    assert len(nut.images) == 0


def test_attack_repeating_known_images_is_stopped_with_warning(art, caplog, monkeypatch):
    def repeat_original(estimator, x):
        for _ in range(10 * STALL_LIMIT):
            estimator.predict(x)
        raise AssertionError("the stalled attack must be stopped")

    _install_fake_attack(monkeypatch, "simba", repeat_original)
    algorithm, nut = make_algorithm("simba", simba_attack="dct")
    with caplog.at_level(logging.WARNING):
        solution = algorithm.search()
    assert "no new image" in caplog.text
    assert "simba_epsilon" in caplog.text
    assert solution.actions == []
    assert len(nut.images) == 0


def test_simba_default_basis_is_px():
    assert ConfigParser().default_params()["simba_attack"] == "px"
