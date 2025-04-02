from tests.e2e.e2e_base import E2EBase


class TestOriginal(E2EBase):
    COLLECT_INFO_PORT = 3410  # Override the default port
    TEST_FOLDER = "output_defdef"

    @staticmethod
    def get_process_image_func():
        # Define the original process_image function
        counter = None

        def process_image(encoded_image: bytes):
            nonlocal counter
            if counter is None:
                counter = 0
            else:
                counter += 1

            prediction_value = 0.001 * counter
            return {
                "predictions": [
                    {"label": "zebra", "score": prediction_value},
                    {"label": "horse", "score": 1 - prediction_value},
                ]
            }

        return process_image

    @staticmethod
    def get_sys_argv():
        return [
            "main.py",
            "--show_plots",
            "false",
            "--nut_port",
            str(TestOriginal.COLLECT_INFO_PORT),  # Use the class variable
            "--snapshot_interval",
            "1"
        ]

    def custom_assertions(self):
        assert self._app_instance.stc.current_fitness_value.value == 0.0