import os
import sys
from typing import Callable, Optional

import pytest

from core.problem.base_module import BaseModule
from core.utils.application import configure_container
from main import OptiAttack


class E2EBase:
    # Class variables that can be overridden by child classes
    COLLECT_INFO_HOST: str = "localhost"
    COLLECT_INFO_PORT: int = 3410
    TEST_FOLDER = "output_default"

    COLLECT_INFO_KWARGS: dict = {}
    _app_instance: Optional[OptiAttack] = None

    @staticmethod
    def get_process_image_func() -> Callable:
        """To be overridden by child classes to provide different process_image implementations"""
        raise NotImplementedError("Child classes must implement get_process_image_func()")

    @staticmethod
    def get_sys_argv() -> list:
        """To be overridden by child classes to provide different command line arguments"""
        raise NotImplementedError("Child classes must implement get_sys_argv()")

    @staticmethod
    def get_collect_info_decorator() -> Callable:
        """Optional: Override to provide a custom collect_info decorator"""
        from client.optiattack_client import collect_info
        return collect_info

    @pytest.fixture
    def container(self):
        # Set arguments for the argparse
        original_argv = sys.argv
        sys.argv = self.get_sys_argv()
        sys.argv.append("--output_dir")
        sys.argv.append(self.TEST_FOLDER)

        # Get the process_image function and decorate it
        process_image_func = self.get_process_image_func()
        collect_info = self.get_collect_info_decorator()

        decorated_func = collect_info(
            host=self.COLLECT_INFO_HOST,
            port=self.COLLECT_INFO_PORT,
            **self.COLLECT_INFO_KWARGS
        )(process_image_func)

        # Make the decorated function available
        globals()['process_image'] = decorated_func

        container = BaseModule()
        config_parser = container.config_parser()
        parsed_args = config_parser.parse_args()
        container.config.override(parsed_args)
        container = configure_container(container)

        yield container

        # Restore original sys.argv
        sys.argv = original_argv

    @pytest.fixture
    def app(self, container) -> OptiAttack:
        """Fixture that provides the app instance and stores it for later access"""
        app = OptiAttack()
        container.wire(modules=[app])
        app.startup()
        self._app_instance = app  # Store the instance
        yield app
        self._app_instance = None  # Clean up

    def test_run_nut(self, app):
        """Base test that runs the application and verifies basic functionality"""
        app.run()
        self.verify_basic_output()

    def verify_basic_output(self):
        """Basic output verification that can be extended or overridden"""
        assert self._app_instance is not None, "App instance not available"

        output_folder = self._app_instance.config.get("output_dir")
        experiments_folder = self._app_instance.config.get("experiment_label")
        seed = self._app_instance.config.get("seed")

        save_folder = f"{output_folder}/{experiments_folder}/{seed}"
        # Check if files exist
        assert os.path.exists(save_folder)
        assert os.path.exists(f"{save_folder}/images")
        assert os.path.exists(f"{save_folder}/statistics")

        # Check if the image is saved
        assert os.path.exists(f"{save_folder}/images/final_image.jpg")
        assert os.path.exists(f"{save_folder}/images/line.png")
        assert os.path.exists(f"{save_folder}/images/matrix_overlay.png")

        # Check if the statistics are saved
        assert os.path.exists(f"{save_folder}/statistics/data.json")

        self.additional_assertions()

    def additional_assertions(self):
        """Override this method in child classes to add custom assertions"""
        pass