"""remote_controller.py - RemoteController class to interact with the NUT server."""

import logging
import requests

from client import constants


class RemoteController:

    """RemoteController class to interact with the NUT server."""

    def __init__(self, config, search_time_controller):
        """Initialize the RemoteController class."""
        # python 3.10 is not supported nested f-string. So,
        # we need to use config['NUT_PORT'] instead of config["NUT_PORT"]
        base_url = (f"http://{config.get('nut_host')}:"
                    f"{config.get('nut_port')}{constants.BASE_PATH}")
        self.NUT_ENDPOINTS = {
            "info": f"{base_url}/infoNUT",
            "run": f"{base_url}/runNUT",
            "stop": f"{base_url}/stopNUT",
            "testResults": f"{base_url}/testResults",
            "newAction": f"{base_url}/newAction"
        }

        # Create a session to keep the connection alive
        self.connection = requests.Session()
        self.search_time_controller = search_time_controller

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
            json_data = image_array.tolist()
            return self.connection.post(self.NUT_ENDPOINTS["run"],
                                        json={"image": json_data}).json()
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
            json_data = image_array.tolist()
            self.search_time_controller.new_individual_evaluation()
            return self.connection.post(self.NUT_ENDPOINTS["newAction"],
                                        json={"image": json_data}).json()
        except requests.exceptions.ConnectionError:
            logging.error("Connection Error")
            raise ConnectionError("Connection Error")
