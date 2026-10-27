import csv

import numpy as np
import pytest

pytest.importorskip("art")

from tests.e2e.e2e_base import E2EBase  # noqa: E402

LABELS = ["zebra", "horse", "dog"]
MAX_EVALUATIONS = 60
SIZE = 32
N_FEATURES = SIZE * SIZE * 3
_RNG = np.random.default_rng(0)
# Linear model on the raw 0-255 pixels: logits = W @ x + b.
W = _RNG.normal(scale=0.77 / N_FEATURES, size=(len(LABELS), N_FEATURES))
B = np.array([0.5, 0.0, -0.5])


def _logits(image):
    return W @ np.asarray(image, dtype=np.float64).ravel() + B


def _softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


def _process_image(encoded_image: bytes, additional_data=None):
    """Probabilities and logits of the linear model, for the uint8 image bytes sent by OptiAttack."""
    logits = _logits(encoded_image)
    probs = _softmax(logits)
    return {
        "predictions": [{"label": label, "score": float(p)} for label, p in zip(LABELS, probs)],
        "logits": [float(v) for v in logits],
    }


def _gradient_fn(image, label, kind):
    """Analytic gradients of the linear model: logit ('class') or softmax cross-entropy ('loss')."""
    index = LABELS.index(label)
    if kind == "class":
        return W[index].reshape(image.shape).astype(np.float32)
    probs = _softmax(_logits(image))
    probs[index] -= 1.0
    return (probs @ W).reshape(image.shape).astype(np.float32)


class WhiteboxE2EBase(E2EBase):
    COLLECT_INFO_KWARGS = {"labels": LABELS, "gradient_fn": _gradient_fn}
    ATTACK = ""

    @staticmethod
    def get_process_image_func():
        return _process_image

    @classmethod
    def get_sys_argv(cls):
        return ["--algorithm", "whitebox", "--whitebox_attack", cls.ATTACK,
                "--max_evaluations", str(MAX_EVALUATIONS),
                "--image_width", str(SIZE), "--image_height", str(SIZE)]

    def additional_assertions(self):
        statistics = self._app_instance.statistics
        with open(f"{statistics.output_dir}/{statistics.statistics_file_name}.csv", newline="") as f:
            row = next(csv.DictReader(f))
        # E2EBase repeats the run with one shared search time controller, so the counters add up over the runs; the
        # first run has a fresh budget and must have asked the NUT for gradients.
        assert int(row["gradient_eval_count"]) > 0
        assert int(row["gradient_eval_count"]) <= int(row["eval_count"])
        assert int(row["eval_count"]) <= MAX_EVALUATIONS
        assert row["config_whitebox_attack"] == self.ATTACK
        assert row["config_algorithm"] == "whitebox"


class TestWhiteboxPgd(WhiteboxE2EBase):
    COLLECT_INFO_PORT = 34120
    TEST_FOLDER = "whitebox_pgd"
    ATTACK = "pgd"


class TestWhiteboxFgsm(WhiteboxE2EBase):
    COLLECT_INFO_PORT = 34121
    TEST_FOLDER = "whitebox_fgsm"
    ATTACK = "fgsm"


class TestWhiteboxCw(WhiteboxE2EBase):
    COLLECT_INFO_PORT = 34122
    TEST_FOLDER = "whitebox_cw"
    ATTACK = "cw"
