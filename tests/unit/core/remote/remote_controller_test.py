from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from client import constants
from client.optiattack_client import collect_info
from main import OptiAttack
from core.problem.base_module import BaseModule

PROCESS_IMAGE_RESPONSE = {
        "predictions": [
            {"label": "zebra", "score": 0.99},
            {"label": "horse", "score": 0.01},
        ]
    }

@collect_info(host=constants.DEFAULT_CONTROLLER_HOST, port=constants.DEFAULT_CONTROLLER_PORT)
def process_image(encoded_image: bytes):
    return PROCESS_IMAGE_RESPONSE


@pytest.fixture
def app():
    container = BaseModule()
    container.unwire()
    container.config.override({"seed": 42, "nut_host": constants.DEFAULT_CONTROLLER_HOST, "nut_port": constants.DEFAULT_CONTROLLER_PORT})

    app = OptiAttack(
        config=container.config(),
        remote_controller=container.remote_controller(),
    )
    yield app

def test_get_info_nut(app):

    response = app.remote_controller.get_nut_info()
    assert response["is_running"] is False

def test_run_nut(app):
    response = app.remote_controller.run_nut()
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

    response = app.remote_controller.new_action(image_array)
    assert response["predictions"] == PROCESS_IMAGE_RESPONSE["predictions"]
    image.close()

def test_get_test_results():
    # TODO: Not implemented and tested
    pass