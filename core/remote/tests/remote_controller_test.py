from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from client import constants
from client.optiattack_client import collect_info
from core.config_parser import Config
from core.remote.remote_controller import RemoteController

PROCESS_IMAGE_RESPONSE = {
        "prediction": [
            {"label": "zebra", "score": 0.99},
            {"label": "horse", "score": 0.01},
        ]
    }

@collect_info(host=constants.DEFAULT_CONTROLLER_HOST, port=constants.DEFAULT_CONTROLLER_PORT)
def process_image(encoded_image: bytes):
    return PROCESS_IMAGE_RESPONSE



config = Config([])
remote = RemoteController()


def test_get_info_nut():
    response = remote.get_nut_info()
    assert response["is_running"] is False

def test_run_nut():
    response = remote.run_nut()
    assert response["is_running"] is True

def test_stop_nut():
    response = remote.stop_nut()
    assert response["is_running"] is False

def test_new_action():
    image_data = BytesIO()
    image = Image.new("RGB", (100, 100), color="red")
    image.save(image_data, format="JPEG")
    image_data.seek(0)

    image_array = np.array(image)

    response = remote.new_action(image_array)
    assert response["prediction"] == PROCESS_IMAGE_RESPONSE["prediction"]
    image.close()

def test_get_test_results():
    # TODO: Not implemented and tested
    pass