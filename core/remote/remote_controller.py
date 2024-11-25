import requests
import logging

class RemoteController:

    def __init__(self, config):

        # python 3.10 is not supported nested f-string. So, we need to use config['NUT_PORT'] instead of config["NUT_PORT"]
        base_url = f"http://{config['NUT_HOST']}:{config['NUT_PORT']}{config['NUT_BASE_PATH']}"
        self.NUT_ENDPOINTS = {
            "info": f"{base_url}/infoNUT",
            "run": f"{base_url}/runNUT",
            "stop": f"{base_url}/stopNUT",
            "testResults": f"{base_url}/testResults",
            "newAction": f"{base_url}/newAction"
        }

        ## Create a session to keep the connection alive
        self.connection = requests.Session()

    def get_nut_info(self):
        try:
            logging.info("Getting NUT info")
            return self.connection.get(self.NUT_ENDPOINTS["info"]).json()
        except requests.exceptions.ConnectionError:
            logging.error("Connection Error")
            return {"error": "Connection Error"}

    def run_nut(self):
        try:
            logging.info("Running NUT")
            return self.connection.post(self.NUT_ENDPOINTS["run"]).json()
        except requests.exceptions.ConnectionError:
            logging.error("Connection Error")
            return {"error": "Connection Error"}

    def stop_nut(self):
        try:
            logging.info("Stopping NUT")
            return self.connection.post(self.NUT_ENDPOINTS["stop"]).json()
        except requests.exceptions.ConnectionError:
            logging.error("Connection Error")
            return {"error": "Connection Error"}

    def get_test_results(self):
        # TODO: Not implemented and tested
        try:
            logging.info("Getting test results")
            return self.connection.get(self.NUT_ENDPOINTS["testResults"]).json()
        except requests.exceptions.ConnectionError:
            logging.error("Connection Error")
            return {"error": "Connection Error"}

    def new_action(self, file):
        try:
            logging.info("Sending new action")
            return self.connection.post(self.NUT_ENDPOINTS["newAction"], files=file).json()
        except requests.exceptions.ConnectionError:
            logging.error("Connection Error")
            return {"error": "Connection Error"}
