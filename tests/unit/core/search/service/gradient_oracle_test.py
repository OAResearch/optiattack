import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.remote.remote_controller import RemoteController
from core.search.phase_controller import PhaseController
from core.search.service.gradient_oracle import GradientOracle
from core.search.service.search_time_controller import SearchTimeController

LABEL = "cat"


def make_oracle(use_gradients=True, capabilities=("scores", "gradients"), gradient=None):
    config = {"use_gradients": use_gradients}
    remote = MagicMock()
    remote.get_nut_info.return_value = {"capabilities": list(capabilities), "labels": [LABEL, "dog"]}
    if gradient is None:
        gradient = np.zeros((4, 5, 3), dtype=np.float32)
    remote.gradient.return_value = gradient
    archive = MagicMock()
    archive.get_original_prediction_results.return_value = SimpleNamespace(max_score=SimpleNamespace(label=LABEL))
    return GradientOracle(config, remote, archive), remote


def test_unavailable_by_default():
    oracle, remote = make_oracle(use_gradients=False)

    assert oracle.available() is False
    remote.get_nut_info.assert_not_called()


def test_unavailable_without_nut_capability():
    oracle, _ = make_oracle(capabilities=("scores",))

    assert oracle.available() is False


def test_available_and_saliency_shape():
    gradient = np.zeros((4, 5, 3), dtype=np.float32)
    gradient[1, 2] = (3.0, 4.0, 0.0)
    oracle, remote = make_oracle(gradient=gradient)

    assert oracle.available() is True
    saliency = oracle.pixel_saliency(np.zeros((4, 5, 3), dtype=np.uint8))

    assert saliency.shape == (4, 5)
    assert saliency.dtype == np.float32
    assert saliency[1, 2] == pytest.approx(5.0)
    assert saliency[0, 0] == 0.0
    assert remote.gradient.call_args[0][1:] == (LABEL, "loss")
    assert remote.get_nut_info.call_count == 1


def test_saliency_counts_one_gradient_query():
    config = {"use_gradients": True, "nut_host": "localhost", "nut_port": 38000, "base_endpoint": "/api/v1",
              "target": None}
    stc = SearchTimeController(config, pc=PhaseController())
    controller = RemoteController(config, stc)
    gradient = np.ones((4, 5, 3), dtype=np.float32)
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"gradient": base64.b64encode(gradient.tobytes()).decode(), "shape": [4, 5, 3]}
    controller.connection = MagicMock()
    controller.connection.post.return_value = response
    controller.get_nut_info = MagicMock(return_value={"capabilities": ["scores", "gradients"], "labels": [LABEL]})
    archive = MagicMock()
    archive.get_original_prediction_results.return_value = SimpleNamespace(max_score=SimpleNamespace(label=LABEL))
    oracle = GradientOracle(config, controller, archive)

    oracle.pixel_saliency(np.zeros((4, 5, 3), dtype=np.uint8))

    assert stc.get_gradient_evaluations() == 1
    assert stc.get_evaluated_individuals() == 1


def test_saliency_raises_when_unavailable():
    oracle, remote = make_oracle(use_gradients=False)

    with pytest.raises(RuntimeError):
        oracle.pixel_saliency(np.zeros((4, 5, 3), dtype=np.uint8))
    remote.gradient.assert_not_called()
