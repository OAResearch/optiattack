import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from io import BytesIO
import base64

from client.core import constants
from client.core.optiattack_client import collect_info

PROCESS_IMAGE_RESPONSE = {
    "predictions": [
        {"label": "zebra", "score": 0.99},
        {"label": "horse", "score": 0.01},
    ]
}


@collect_info(host=constants.DEFAULT_CONTROLLER_HOST,
              port=constants.DEFAULT_CONTROLLER_PORT)
def process_image(image_array: np.ndarray):
    return PROCESS_IMAGE_RESPONSE


@pytest.fixture(scope="module")
def setup_test_app():
    client = TestClient(process_image.app)
    return client


def test_fastapi_server(setup_test_app):
    response = setup_test_app.get("/")
    assert response.status_code == 200


def test_stop_nut_endpoint(setup_test_app):
    response = setup_test_app.post(constants.STOP_NUT_PATH)
    assert response.status_code == 200
    assert response.json()["is_running"] is False


def test_info_nut_endpoint(setup_test_app):
    response = setup_test_app.get(constants.INFO_NUT_PATH)
    assert response.status_code == 200
    assert response.json()["is_running"] is False


app = process_image.app
client = TestClient(app)


def get_test_image():
    image_data = BytesIO()
    image = Image.new("RGB", (100, 100), color="red")
    image.save(image_data, format="JPEG")
    image_data.seek(0)
    return image


def test_run_nut_endpoint(setup_test_app):
    image = get_test_image()

    json_data = base64.b64encode(np.array(image)).decode()
    body = {"image": json_data}

    response = setup_test_app.post(constants.RUN_NUT_PATH, json=body)
    assert response.status_code == 200
    assert response.json()["is_running"] is True


def test_new_action_endpoint():
    image = get_test_image()

    json_data = base64.b64encode(np.array(image)).decode()
    body = {"image": json_data}

    response = client.post(constants.NEW_ACTION, json=body)

    assert response.status_code == 200
    assert response.json() == PROCESS_IMAGE_RESPONSE
