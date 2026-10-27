import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from client.core import constants
from client.core.optiattack_client import collect_info

LABELS = ["zebra", "horse", "cat"]
H, W, C = 2, 2, 3
WEIGHTS = np.random.RandomState(0).randn(len(LABELS), H * W * C).astype(np.float32)
SEEN = []


def _logits(image):
    return WEIGHTS @ image.reshape(-1).astype(np.float32)


def linear_gradient(image, label, kind):
    """Analytic gradient of a numpy linear model."""
    SEEN.append((image.dtype, image.shape, label, kind))
    k = LABELS.index(label)
    if kind == "class":
        return WEIGHTS[k].reshape(H, W, C).copy()
    logits = _logits(image)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    p[k] -= 1.0
    return (p @ WEIGHTS).reshape(H, W, C).astype(np.float32)


def wrong_shape_gradient(image, label, kind):
    return np.zeros((H, W), dtype=np.float32)


@collect_info(host=constants.DEFAULT_CONTROLLER_HOST, port=38120, labels=LABELS, gradient_fn=linear_gradient)
def wb_model(data, additional_data=None):
    logits = _logits(np.asarray(data, dtype=np.float32))
    return {"predictions": [0.5, 0.3, 0.2], "logits": logits.tolist()}


@collect_info(host=constants.DEFAULT_CONTROLLER_HOST, port=38121, labels=LABELS)
def plain_model(data, additional_data=None):
    return {"predictions": [0.5, 0.3, 0.2]}


@collect_info(host=constants.DEFAULT_CONTROLLER_HOST, port=38122, labels=LABELS, gradient_fn=wrong_shape_gradient)
def bad_shape_model(data, additional_data=None):
    return {"predictions": [0.5, 0.3, 0.2]}


def _b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode()


def _body(image, label="horse", kind="loss", shape=None):
    return {"image": _b64(image), "shape": list(shape or image.shape), "label": label, "kind": kind}


def _image():
    return (np.random.RandomState(1).rand(H, W, C) * 255).astype(np.float32)


def test_capabilities_include_gradients():
    data = TestClient(wb_model.app).get(constants.INFO_NUT_PATH).json()
    assert data["capabilities"] == ["scores", "gradients"]
    assert data["labels"] == LABELS


def test_client_without_gradient_fn_unchanged():
    data = TestClient(plain_model.app).get(constants.INFO_NUT_PATH).json()
    assert data["capabilities"] == ["scores"]


def test_gradient_endpoint_roundtrip():
    client = TestClient(wb_model.app)
    image = _image()
    for kind in ("loss", "class"):
        resp = client.post(constants.GRADIENT_NUT, json=_body(image, "horse", kind))
        assert resp.status_code == 200
        out = resp.json()
        assert out["shape"] == [H, W, C]
        grad = np.frombuffer(base64.b64decode(out["gradient"]), dtype=np.float32).reshape(H, W, C)
        np.testing.assert_allclose(grad, linear_gradient(image, "horse", kind), rtol=1e-6)
    dtype, shape, label, kind = SEEN[-2]
    assert dtype == np.float32 and shape == (H, W, C) and label == "horse" and kind == "class"


def test_gradient_endpoint_rejects_unknown_kind_and_label():
    client = TestClient(wb_model.app)
    image = _image()
    resp = client.post(constants.GRADIENT_NUT, json=_body(image, kind="bogus"))
    assert resp.status_code == 400 and "kind" in resp.json()["detail"]
    resp = client.post(constants.GRADIENT_NUT, json=_body(image, label="dog"))
    assert resp.status_code == 400 and "label" in resp.json()["detail"]


def test_gradient_endpoint_rejects_bad_image_payload():
    client = TestClient(wb_model.app)
    image = _image()
    resp = client.post(constants.GRADIENT_NUT, json=_body(image, shape=(H, W, C + 1)))
    assert resp.status_code == 400 and "shape" in resp.json()["detail"]
    body = _body(image)
    body["image"] = "!!not base64!!"
    assert client.post(constants.GRADIENT_NUT, json=body).status_code == 400
    body = _body(image)
    body["image"] = base64.b64encode(b"abc").decode()
    assert client.post(constants.GRADIENT_NUT, json=body).status_code == 400


def test_gradient_endpoint_without_gradient_fn_is_400():
    resp = TestClient(plain_model.app).post(constants.GRADIENT_NUT, json=_body(_image()))
    assert resp.status_code == 400
    assert "gradient" in resp.json()["detail"]


def test_gradient_endpoint_rejects_wrong_returned_shape():
    resp = TestClient(bad_shape_model.app).post(constants.GRADIENT_NUT, json=_body(_image()))
    assert resp.status_code == 400 and "shape" in resp.json()["detail"]


def test_logits_pass_through_new_action():
    client = TestClient(wb_model.app)
    raw = np.arange(H * W * C, dtype=np.uint8)
    body = {"image": base64.b64encode(raw.tobytes()).decode()}
    out = client.post(constants.NEW_ACTION, json=body).json()
    assert out["predictions"] == [0.5, 0.3, 0.2]
    expected = _logits(raw.astype(np.float32))
    np.testing.assert_allclose(out["logits"], expected, rtol=1e-5)
    assert len(out["logits"]) == len(LABELS)


def test_run_nut_stores_logits():
    client = TestClient(wb_model.app)
    raw = np.arange(H * W * C, dtype=np.uint8)
    payload = {"image": base64.b64encode(raw.tobytes()).decode()}
    state = client.post(constants.RUN_NUT_PATH, json=payload).json()
    assert len(state["logits"]) == len(LABELS)
    plain_state = TestClient(plain_model.app).post(constants.RUN_NUT_PATH, json=payload).json()
    assert "logits" not in plain_state


def test_gradient_fn_without_labels_fails_at_decoration_time():
    for missing in (None, []):
        with pytest.raises(ValueError, match="labels"):
            collect_info(host=constants.DEFAULT_CONTROLLER_HOST, port=38123, labels=missing,
                         gradient_fn=linear_gradient)
