import base64
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.remote.remote_controller import NutGradientError, RemoteController
from core.search.algorithms.baseline.nut_interface import InvalidGradientError
from core.utils.nut_request import NutRequest

CONFIG = {"nut_host": "localhost", "nut_port": 38000, "base_endpoint": "/api/v1", "target": None}


def make_controller(status_code=200, payload=None):
    stc = MagicMock()
    controller = RemoteController(CONFIG, stc)
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    controller.connection = MagicMock()
    controller.connection.post.return_value = response
    return controller, stc


def encode(array):
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.float32).tobytes()).decode()


def test_gradient_endpoint_is_registered():
    controller, _ = make_controller()
    assert controller.NUT_ENDPOINTS["gradient"] == "http://localhost:38000/api/v1/gradientNut"


def test_gradient_posts_float32_and_counts_both_counters():
    gradient = np.arange(12, dtype=np.float32).reshape(2, 2, 3) - 5.5
    controller, stc = make_controller(payload={"gradient": encode(gradient), "shape": [2, 2, 3]})
    image = np.full((2, 2, 3), 10.25, dtype=np.float64)  # not uint8: the gradient is taken at the float point

    result = controller.gradient(image, "cat", "loss")

    url = controller.connection.post.call_args[0][0]
    body = controller.connection.post.call_args[1]["json"]
    assert url == "http://localhost:38000/api/v1/gradientNut"
    assert body["shape"] == [2, 2, 3]
    assert body["label"] == "cat"
    assert body["kind"] == "loss"
    sent = np.frombuffer(base64.b64decode(body["image"]), dtype=np.float32).reshape(2, 2, 3)
    np.testing.assert_array_equal(sent, np.full((2, 2, 3), 10.25, dtype=np.float32))
    stc.new_individual_evaluation.assert_called_once()
    stc.new_gradient_evaluation.assert_called_once()
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, gradient)


def test_gradient_http_400_becomes_clear_error():
    controller, _ = make_controller(status_code=400, payload={"detail": "this NUT does not provide gradient_fn"})
    with pytest.raises(NutGradientError, match="gradient_fn") as info:
        controller.gradient(np.zeros((2, 2, 3)), "dog", "class")
    assert "dog" in str(info.value) and "class" in str(info.value) and "400" in str(info.value)


def test_gradient_with_inconsistent_byte_count_raises_invalid_gradient():
    controller, _ = make_controller(payload={"gradient": encode(np.zeros(5)), "shape": [2, 2, 3]})
    with pytest.raises(InvalidGradientError, match="cat"):
        controller.gradient(np.zeros((2, 2, 3)), "cat", "loss")


def test_nut_request_keeps_optional_logits():
    predictions = [{"label": "cat", "score": 0.6}, {"label": "dog", "score": 0.4}]
    assert NutRequest({"predictions": predictions}).logits is None
    assert NutRequest({"predictions": predictions, "logits": [1.5, -2.0]}).logits == [1.5, -2.0]
