import numpy as np
from fastapi.testclient import TestClient

from client.core import constants
from client.core.optiattack_client import collect_info


@collect_info(host=constants.DEFAULT_CONTROLLER_HOST, port=38110, labels=["zebra", "horse"])
def process_with_labels(image_array: np.ndarray, additional_data=None):
    return {"predictions": [0.9, 0.1]}


@collect_info(host=constants.DEFAULT_CONTROLLER_HOST, port=38111)
def process_without_labels(image_array: np.ndarray, additional_data=None):
    return {"predictions": [0.9, 0.1]}


def test_info_nut_reports_labels_and_capabilities():
    client = TestClient(process_with_labels.app)
    data = client.get(constants.INFO_NUT_PATH).json()
    assert data["labels"] == ["zebra", "horse"]
    assert data["capabilities"] == ["scores"]


def test_info_nut_without_labels_reports_null():
    client = TestClient(process_without_labels.app)
    data = client.get(constants.INFO_NUT_PATH).json()
    assert data["labels"] is None
    assert data["capabilities"] == ["scores"]
