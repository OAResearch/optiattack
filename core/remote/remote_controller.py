"""remote_controller.py - RemoteController class to interact with the NUT server."""
import base64
import logging
import numpy as np
import requests

from core.search.algorithms.baseline.nut_interface import InvalidGradientError
from core.utils.nut_request import NutRequest


class NutGradientError(RuntimeError):

    """Raised when the NUT rejects a gradient request (e.g. it has no gradient_fn)."""


class RemoteController:

    """RemoteController class to interact with the NUT server."""

    def __init__(self, config, stc):
        """Initialize the RemoteController class."""
        # python 3.10 is not supported nested f-string. So,
        # we need to use config['NUT_PORT'] instead of config["NUT_PORT"]
        base_url = (f"http://{config.get('nut_host')}:"
                    f"{config.get('nut_port')}{config.get('base_endpoint')}")
        self.NUT_ENDPOINTS = {
            "info": f"{base_url}/infoNUT",
            "run": f"{base_url}/runNUT",
            "stop": f"{base_url}/stopNUT",
            "testResults": f"{base_url}/testResults",
            "newAction": f"{base_url}/newAction",
            "gradient": f"{base_url}/gradientNut"
        }

        # Create a session to keep the connection alive
        self.connection = requests.Session()
        self.stc = stc
        self.target = config.get("target")

    def get_nut_info(self):
        """Get NUT info."""
        try:
            logging.info("Getting NUT info")
            return self.connection.get(self.NUT_ENDPOINTS["info"]).json()
        except requests.exceptions.ConnectionError:
            logging.error("Connection Error")
            raise ConnectionError("Connection Error")

    def run_nut(self, image_array):
        """Run NUT."""
        try:
            logging.info("Running NUT. Sending image to NUT for testing...")
            json_data = base64.b64encode(image_array).decode()
            return self.connection.post(self.NUT_ENDPOINTS["run"],
                                        json={"image": json_data, "target": self.target}).json()
        except requests.exceptions.ConnectionError:
            logging.error("Connection Error")
            raise ConnectionError("Connection Error")

    def stop_nut(self):
        """Stop NUT."""
        try:
            logging.info("Stopping NUT")
            return self.connection.post(self.NUT_ENDPOINTS["stop"]).json()
        except requests.exceptions.ConnectionError:
            logging.error("Connection Error")
            raise ConnectionError("Connection Error")

    def get_test_results(self):
        """Get test results."""
        # TODO: Not implemented and tested
        try:
            logging.info("Getting test results")
            return (self.connection.get(self.NUT_ENDPOINTS["testResults"])
                    .json())
        except requests.exceptions.ConnectionError:
            logging.error("Connection Error")
            raise ConnectionError("Connection Error")

    def new_action(self, image_array):
        """Send new action."""
        try:
            logging.info("Sending new action")
            json_data = base64.b64encode(image_array).decode()
            self.stc.new_individual_evaluation()
            response = self.connection.post(self.NUT_ENDPOINTS["newAction"],
                                            json={"image": json_data}).json()
            nut_request = NutRequest(response, self.target)
            return nut_request

        except requests.exceptions.ConnectionError:
            logging.error("Connection Error")
            raise ConnectionError("Connection Error")

    def gradient(self, image_array, label: str, kind: str) -> np.ndarray:
        """Asks the NUT for the gradient of `kind` ("loss" or "class") for `label` at the float32 image."""
        image = np.ascontiguousarray(image_array, dtype=np.float32)
        body = {"image": base64.b64encode(image.tobytes()).decode(), "shape": list(image.shape),
                "label": label, "kind": kind}
        try:
            logging.info("Requesting %s gradient for label %s", kind, label)
            # A gradient query costs one evaluation of the shared budget and is also counted on its own
            self.stc.new_individual_evaluation()
            self.stc.new_gradient_evaluation()
            response = self.connection.post(self.NUT_ENDPOINTS["gradient"], json=body)
        except requests.exceptions.ConnectionError:
            logging.error("Connection Error")
            raise ConnectionError("Connection Error")
        if response.status_code >= 400:
            raise NutGradientError(
                "The NUT rejected the '%s' gradient request for label '%s' (HTTP %s): %s"
                % (kind, label, response.status_code, self._error_detail(response)))
        payload = response.json()
        values = np.frombuffer(base64.b64decode(payload["gradient"]), dtype=np.float32)
        shape = tuple(payload.get("shape") or image.shape)
        if values.size != int(np.prod(shape)):
            raise InvalidGradientError(
                "The NUT returned %d gradient values for label '%s' (kind '%s') but declared shape %s."
                % (values.size, label, kind, shape))
        return values.reshape(shape).copy()

    @staticmethod
    def _error_detail(response) -> str:
        """Extracts the 'detail' message of an error response, falling back to its text."""
        try:
            return str(response.json().get("detail"))
        except Exception:
            return str(getattr(response, "text", ""))
