import csv

import numpy as np
import pytest

pytest.importorskip("art")

from tests.e2e.e2e_base import E2EBase  # noqa: E402

LABELS = ["zebra", "horse", "dog"]
MAX_EVALUATIONS = 60


def _process_image(encoded_image: bytes, additional_data=None):
    """Score depends on the mean brightness of the raw image bytes so attacks can make progress."""
    zebra = 0.2 + 0.6 * float(np.mean(encoded_image, dtype=np.float64)) / 255.0
    rest = 1.0 - zebra
    return {
        "predictions": [
            {"label": "zebra", "score": zebra},
            {"label": "horse", "score": rest * 0.75},
            {"label": "dog", "score": rest * 0.25},
        ]
    }


class BaselineE2EBase(E2EBase):
    COLLECT_INFO_KWARGS = {"labels": LABELS}
    ATTACK = ""
    EXTRA_ARGV: list = []

    @staticmethod
    def get_process_image_func():
        return _process_image

    @classmethod
    def get_sys_argv(cls):
        return ["--algorithm", "baseline", "--baseline_attack", cls.ATTACK,
                "--max_evaluations", str(MAX_EVALUATIONS),
                "--image_width", "32", "--image_height", "32"] + list(cls.EXTRA_ARGV)

    def additional_assertions(self):
        statistics = self._app_instance.statistics
        with open(f"{statistics.output_dir}/{statistics.statistics_file_name}.csv", newline="") as f:
            row = next(csv.DictReader(f))
        # E2EBase repeats the run with one shared search time controller, so only the first run starts with a
        # fresh budget; it must have queried the NUT.
        self._runs = getattr(self, "_runs", 0) + 1
        if self._runs == 1:
            assert int(row["eval_count"]) > 0
        assert int(row["eval_count"]) <= MAX_EVALUATIONS
        assert "first_success_eval" in row
        assert row["config_baseline_attack"] == self.ATTACK
        assert row["config_algorithm"] == "baseline"


class TestBaselineSquare(BaselineE2EBase):
    COLLECT_INFO_PORT = 34110
    TEST_FOLDER = "baseline_square"
    ATTACK = "square"


class TestBaselineHsj(BaselineE2EBase):
    COLLECT_INFO_PORT = 34111
    TEST_FOLDER = "baseline_hsj"
    ATTACK = "hsj"


class TestBaselineSimba(BaselineE2EBase):
    COLLECT_INFO_PORT = 34112
    TEST_FOLDER = "baseline_simba"
    ATTACK = "simba"
    EXTRA_ARGV = ["--simba_attack", "px"]


class TestBaselinePixel(BaselineE2EBase):
    COLLECT_INFO_PORT = 34113
    TEST_FOLDER = "baseline_pixel"
    ATTACK = "pixel"
