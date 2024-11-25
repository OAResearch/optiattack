import pytest
from PIL import Image
from fastapi.testclient import TestClient
from io import BytesIO


from client import constants
from client.optiattack_client import collect_info

PROCESS_IMAGE_RESPONSE = {
    "prediction": [
        {"label": "zebra", "score": 0.99},
        {"label": "horse", "score": 0.01},
    ]
}

@collect_info(host=constants.DEFAULT_CONTROLLER_HOST, port=constants.DEFAULT_CONTROLLER_PORT)
def process_image(encoded_image: bytes):
    return PROCESS_IMAGE_RESPONSE


@pytest.fixture(scope="module")
def setup_test_app():
    client = TestClient(process_image.app)  # Decorator içindeki `app`'e erişiyoruz.
    return client


def test_fastapi_server(setup_test_app):
    response = setup_test_app.get("/")
    assert response.status_code == 200


def test_run_nut_endpoint(setup_test_app):
    response = setup_test_app.post(constants.RUN_NUT_PATH)
    assert response.status_code == 200
    assert response.json()["is_running"] is True


def test_stop_nut_endpoint(setup_test_app):
    response = setup_test_app.post(constants.STOP_NUT_PATH)
    assert response.status_code == 200
    assert response.json()["is_running"] is False


def test_info_nut_endpoint(setup_test_app):
    response = setup_test_app.get(constants.INFO_NUT_PATH)
    assert response.status_code == 200
    assert response.json()["is_running"] is False
# FastAPI uygulamasını doğrudan alıyoruz.
app = process_image.app
client = TestClient(app)

def test_new_action_endpoint():
    # Mock bir resim oluşturup yükleyelim.
    image_data = BytesIO()
    image = Image.new("RGB", (100, 100), color="red")
    image.save(image_data, format="JPEG")
    image_data.seek(0)

    # Dosya yükleme için senkron istemciyi kullanıyoruz.
    files = {"image": ("test.jpg", image_data, "image/jpeg")}
    response = client.post(constants.NEW_ACTION, files=files)

    assert response.status_code == 200
    assert response.json() == PROCESS_IMAGE_RESPONSE