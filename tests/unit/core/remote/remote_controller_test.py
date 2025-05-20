from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from client.core.optiattack_client import collect_info
from main import OptiAttack
from core.problem.base_module import BaseModule

RESPONSE = {
    "predictions": [
        {"label": "zebra", "score": 0.93},
        {"label": "horse", "score": 0.07},
    ]
}

HOST = "localhost"
PORT = 38030

@collect_info(host=HOST, port=PORT)
def process_image(encoded_image: bytes, additional_data=None):
    return RESPONSE


@pytest.fixture
def app():
    container = BaseModule()
    container.unwire()
    container.config.override(
        {"seed": 42, "nut_host": HOST, "nut_port": PORT,
         "base_endpoint": "/api/v1"})

    app = OptiAttack(
        config=container.config(),
        remote_controller=container.remote_controller(),
    )
    yield app


def test_run_nut(app):
    image_data = BytesIO()
    image = Image.new("RGB", (100, 100), color="red")
    image.save(image_data, format="JPEG")
    image_data.seek(0)

    image_array = np.array(image)

    response = app.remote_controller.run_nut(image_array)
    assert response["is_running"] is True


def test_stop_nut(app):
    response = app.remote_controller.stop_nut()
    assert response["is_running"] is False


def test_new_action(app):
    image_data = BytesIO()
    image = Image.new("RGB", (100, 100), color="red")
    image.save(image_data, format="JPEG")
    image_data.seek(0)

    image_array = np.array(image)
    image.close()


    response = app.remote_controller.new_action(image_array)
    assert response.predictions[0].label == RESPONSE["predictions"][0]["label"]
    assert response.predictions[0].value == RESPONSE["predictions"][0]["score"]
    assert response.predictions[1].label == RESPONSE["predictions"][1]["label"]
    assert response.predictions[1].value == RESPONSE["predictions"][1]["score"]
    assert response.max_score.value == RESPONSE["predictions"][0]["score"]
    assert response.max_score.label == RESPONSE["predictions"][0]["label"]

    assert response.second_max_score.value == RESPONSE["predictions"][1]["score"]
    assert response.second_max_score.label == RESPONSE["predictions"][1]["label"]


def test_get_test_results():
    # TODO: Not implemented and tested
    pass
